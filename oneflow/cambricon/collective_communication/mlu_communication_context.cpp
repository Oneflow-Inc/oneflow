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
#include "oneflow/cambricon/collective_communication/mlu_communication_context.h"
#include "oneflow/cambricon/collective_communication/eager_cncl_comm_manager.h"

namespace oneflow {

namespace ccl {

void MluCommunicationContext::Init(Symbol<ParallelDesc> parallel_desc) {
  std::set<std::pair<int64_t, int64_t>> device_set;
  FOR_RANGE(int64_t, parallel_id, 0, parallel_desc->parallel_num()) {
    int64_t machine_id = CHECK_JUST(parallel_desc->MachineId4ParallelId(parallel_id));
    int64_t device_id = CHECK_JUST(parallel_desc->DeviceId4ParallelId(parallel_id));
    device_set.emplace(std::make_pair(machine_id, device_id));
    rank2cncl_index_.emplace(machine_id, parallel_id);
  }
  cncl_comm_ = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get())
                   ->As<EagerCnclCommMgr>()
                   ->GetCommForDevice(device_set);
}

REGISTER_COLLECTIVE_COMMUNICATION_COMMUNICATOR(DeviceType::kMLU, MluCommunicationContext);

}  // namespace ccl

}  // namespace oneflow
