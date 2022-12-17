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
#include "oneflow/core/control/rank_info_bootstrap_client.h"

namespace oneflow {

namespace {
#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)
}  // namespace

RankInfoBootstrapClient::~RankInfoBootstrapClient() { StopHeartbeat(); }

void RankInfoBootstrapClient::StopHeartbeat() {
  bool already_stopped = false;
  {
    std::unique_lock<std::mutex> lck(heartbeat_thread_mutex_);
    already_stopped = heartbeat_thread_stop_;
    heartbeat_thread_stop_ = true;
    heartbeat_thread_cv_.notify_all();
  }
  if (!already_stopped) { heartbeat_thread_.join(); }
}

RankInfoBootstrapClient::RankInfoBootstrapClient(const BootstrapConf& bootstrap_conf) {
  stubs_.reserve(bootstrap_conf.world_size());
  const auto& master_addr = bootstrap_conf.master_addr();
  const std::string& host = master_addr.host() + ":" + std::to_string(master_addr.port());
  stubs_.emplace_back(CtrlService::NewStub(host));
  LoadServerRequest request;
  request.set_addr(master_addr.host());
  request.set_rank(bootstrap_conf.rank());
  LoadServer(request, stubs_[0].get());

  heartbeat_thread_ = std::thread([this]() {
    std::mt19937 gen(NewRandomSeed());
    std::uniform_int_distribution<int32_t> sleep_second_dis(7, 13);
    LoadServerRequest request;
    LoadServerResponse response;
    while (true) {
      const auto wait_duration = std::chrono::seconds(sleep_second_dis(gen));
      {
        std::unique_lock<std::mutex> lck(heartbeat_thread_mutex_);
        const bool stopped = heartbeat_thread_cv_.wait_for(
            lck, wait_duration, [&]() { return heartbeat_thread_stop_; });
        if (stopped) { break; }
      }
      for (size_t i = 0; i < GetStubSize(); ++i) {
        grpc::ClientContext client_ctx;
        GRPC_CHECK(
            GetStubAt(i)->CallMethod<CtrlMethod::kLoadServer>(&client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
    }
  });
}  // namespace oneflow

}  // namespace oneflow
