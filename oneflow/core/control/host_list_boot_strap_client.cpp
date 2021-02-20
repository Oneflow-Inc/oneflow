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
#include "oneflow/core/control/host_list_boot_strap_client.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

namespace {

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)
}

HostListBootStrapClient::HostListBootStrapClient() {
  stubs_.reserve(Global<EnvDesc>::Get()->TotalMachineNum());
  int32_t port = -1;
  std::string addr = "";
  for (int64_t i = 0; i < Global<EnvDesc>::Get()->TotalMachineNum(); ++i) {
    const Machine& mchn = Global<EnvDesc>::Get()->machine(i);
    port = (mchn.ctrl_port_agent() != -1) ? (mchn.ctrl_port_agent())
                                          : Global<EnvDesc>::Get()->ctrl_port();
    addr = mchn.addr() + ":" + std::to_string(port);
    stubs_.push_back(CtrlService::NewStub(addr));
    LoadServer(mchn.addr(), stubs_[i].get());
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
        request.set_addr(Global<EnvDesc>::Get()->machine(i).addr());
        GRPC_CHECK(stubs_[i]->CallMethod<CtrlMethod::kLoadServer>(&client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
      std::this_thread::sleep_for(std::chrono::seconds(sleep_second_dis(gen)));
    }
  });
}

}  // namespace oneflow
