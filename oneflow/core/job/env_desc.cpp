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
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

size_t EnvDesc::TotalMachineNum() const {
  if (env_proto_.has_ctrl_bootstrap_conf()) {
    return env_proto_.ctrl_bootstrap_conf().world_size();
  } else {
    return env_proto_.machine().size();
  }
}

int64_t EnvDesc::GetMachineId(const std::string& addr) const {
  int64_t machine_id = -1;
  int64_t machine_num = env_proto_.machine_size();
  FOR_RANGE(int64_t, i, 0, machine_num) {
    if (addr == env_proto_.machine(i).addr()) {
      machine_id = i;
      break;
    }
  }
  CHECK_GE(machine_id, 0);
  CHECK_LT(machine_id, machine_num);
  return machine_id;
}

}  // namespace oneflow
