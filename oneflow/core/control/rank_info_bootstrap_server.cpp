/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/control/rank_info_bootstrap_server.h"
#include <cstdio>
#include "grpc/grpc_posix.h"

namespace oneflow {

namespace {

std::string GetHostFromUri(const std::string& uri) {
  size_t first_delimiter_pos = uri.find(":");
  CHECK_NE(first_delimiter_pos, std::string::npos);
  const std::string& protocol_family = uri.substr(0, first_delimiter_pos);
  CHECK_EQ(protocol_family, "ipv4");
  size_t second_delimiter_pos = uri.rfind(":");
  return uri.substr(first_delimiter_pos + 1, second_delimiter_pos - first_delimiter_pos - 1);
}

int64_t rpc_bootsrtap_server_check_seconds() {
  static const int64_t rpc_bootsrtap_server_check_seconds =
      ParseIntegerFromEnv("ONEFLOW_RPC_BOOTSTRAP_SERVER_CHECK_SECONDS", 60);
  return rpc_bootsrtap_server_check_seconds;
}

}  // namespace

RankInfoBootstrapServer::RankInfoBootstrapServer(const BootstrapConf& bootstrap_conf)
    : BootstrapServer(), port_(0), world_size_(bootstrap_conf.world_size()) {
  Init();
  int p = (bootstrap_conf.rank() == 0 ? bootstrap_conf.master_addr().port() : 0);
  grpc::ServerBuilder server_builder;
  server_builder.SetMaxMessageSize(INT_MAX);
  server_builder.AddListeningPort("0.0.0.0:" + std::to_string(p), grpc::InsecureServerCredentials(),
                                  &port_);
  grpc_service_.reset(new CtrlService::AsyncService);
  server_builder.RegisterService(grpc_service_.get());
  cq_ = server_builder.AddCompletionQueue();
  grpc_server_ = server_builder.BuildAndStart();
  if (bootstrap_conf.rank() == 0) { CHECK_EQ(p, port()) << "Port " << p << " is unavailable"; }
  LOG(INFO) << "RankInfoBootstrapServer listening on "
            << "0.0.0.0:" + std::to_string(port());
  loop_thread_ = std::thread(&RankInfoBootstrapServer::HandleRpcs, this);
  if (bootstrap_conf.rank() == 0) {
    rank2host_ = std::make_shared<std::vector<std::string>>(1);
    check_thread_ = std::thread(&RankInfoBootstrapServer::CheckServerRpcs, this);
  }
}

void RankInfoBootstrapServer::CheckServerRpcs() {
  bool status_ok = false;
  size_t timeout_seconds = rpc_bootsrtap_server_check_seconds();
  while (timeout_seconds > 0) {
    CHECK(rank2host_->size() <= world_size_);
    if (rank2host_->size() == world_size_) {
      status_ok = true;
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      timeout_seconds -= 1;
    }
  }
  if (!status_ok) {
    LOG(WARNING)
        << "BootstrapServer not ready, other ranks' rpc server not initialized successfully!"
        << ", world size: " << world_size_ << ", ready ranks: " << rank2host_->size();
    LOG(FATAL) << "Grpc failed.";
  }
}

Maybe<const std::vector<std::string>&> RankInfoBootstrapServer::rank2host() const {
  CHECK_NOTNULL(rank2host_.get());
  return *rank2host_;
}

void RankInfoBootstrapServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) {
  int64_t rank = call->request().rank();
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size_);
  rank2host_->at(rank) = GetHostFromUri(call->server_ctx().peer());
  call->SendResponse();
  EnqueueRequest<CtrlMethod::kLoadServer>();
}

}  // namespace oneflow
