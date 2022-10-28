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
#include "oneflow/user/kernels/collective_communication/cpu/cpu_communication_context.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace ccl {

void CpuCommunicationContext::Init(Symbol<ParallelDesc> parallel_desc) {
  parallel_desc_ = parallel_desc;
}

REGISTER_COLLECTIVE_COMMUNICATION_COMMUNICATOR(DeviceType::kCPU, CpuCommunicationContext);

}  // namespace ccl

}  // namespace oneflow
