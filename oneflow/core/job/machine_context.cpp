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
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<ResourceDesc, ForSession>::Get()->machine(machine_id);
  int32_t ctrl_port = (mchn.ctrl_port_agent() != -1) ? (mchn.ctrl_port_agent())
                                                     : Global<EnvDesc>::Get()->ctrl_port();
  return mchn.addr() + ":" + std::to_string(ctrl_port);
}

MachineCtx::MachineCtx(int64_t this_mchn_id) : this_machine_id_(this_mchn_id) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
