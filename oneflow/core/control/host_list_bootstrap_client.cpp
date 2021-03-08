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
#include "oneflow/core/control/host_list_bootstrap_client.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

HostListBootstrapClient::HostListBootstrapClient(const EnvDesc& env_desc) {
  stubs_.reserve(env_desc.TotalMachineNum());
  int32_t port = -1;
  std::string addr = "";
  for (int64_t i = 0; i < env_desc.TotalMachineNum(); ++i) {
    const Machine& mchn = env_desc.machine(i);
    port = (mchn.ctrl_port_agent() != -1) ? (mchn.ctrl_port_agent()) : env_desc.ctrl_port();
    addr = mchn.addr() + ":" + std::to_string(port);
    stubs_.push_back(CtrlService::NewStub(addr));
    LoadServer(mchn.addr(), stubs_[i].get());
  }
}

}  // namespace oneflow
