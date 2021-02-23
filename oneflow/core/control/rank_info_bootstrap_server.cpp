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
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/job/profiler.h"
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
}

Maybe<const std::vector<std::string>&> RankInfoBootstrapServer::rank2host() const {
  CHECK_NOTNULL(rank2host_.get());
  return *rank2host_;
}

void RankInfoBootstrapServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) {
  int64_t rank = call->request().rank();
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size_);
  if (!rank2host_) { rank2host_ = std::make_shared<std::vector<std::string>>(world_size_); }
  rank2host_->at(rank) = GetHostFromUri(call->server_ctx().peer());
  call->SendResponse();
  EnqueueRequest<CtrlMethod::kLoadServer>();
}

}  // namespace oneflow
