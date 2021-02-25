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
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

namespace {

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)
}  // namespace

CtrlClient::~CtrlClient() {
  {
    std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
    need_heartbeat_thread_stop_ = true;
  }
  heartbeat_thread_.join();
}

CtrlClient::CtrlClient(const ProcessCtx& process_ctx) : process_ctx_(process_ctx) {
  stubs_.reserve(process_ctx.ctrl_addr_size());
  for (int64_t i = 0; i < process_ctx.ctrl_addr_size(); ++i) {
    const Address& address = process_ctx.ctrl_addr(i);
    stubs_.push_back(CtrlService::NewStub(address.host() + ":" + std::to_string(address.port())));
    LoadServer(address.host(), stubs_[i].get());
  }
  need_heartbeat_thread_stop_ = false;
  heartbeat_thread_ = std::thread([this]() {
    std::mt19937 gen(NewRandomSeed());
    std::uniform_int_distribution<int32_t> sleep_second_dis(7, 13);
    LoadServerRequest request;
    LoadServerResponse response;
    while (true) {
      {
        std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
        if (need_heartbeat_thread_stop_) { break; }
      }
      for (size_t i = 0; i < stubs_.size(); ++i) {
        grpc::ClientContext client_ctx;
        request.set_addr(this->process_ctx().ctrl_addr(i).host());
        GRPC_CHECK(stubs_[i]->CallMethod<CtrlMethod::kLoadServer>(&client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
      std::this_thread::sleep_for(std::chrono::seconds(sleep_second_dis(gen)));
    }
  });
}

}  // namespace oneflow
