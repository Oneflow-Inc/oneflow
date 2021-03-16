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
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace eager {

Maybe<void> BlobObject::CheckMemCase(const ParallelDesc& parallel_desc, int64_t machine_id) const {
  CHECK_OR_RETURN(parallel_desc.HasMachineId(machine_id))
      << "ParallelDesc does not contain machine_id: " << machine_id;
  const std::string device_tag = *JUST(DeviceTag4DeviceType(parallel_desc.device_type()));
  if (parallel_desc.device_type() == DeviceType::kCPU) {
    CHECK_OR_RETURN(this->mem_case_->has_host_mem())
        << "DeviceType: " << device_tag
        << " not match MemoryCase: " << this->mem_case_->host_mem().DebugString();
  } else if (parallel_desc.device_type() == DeviceType::kGPU) {
    CHECK_OR_RETURN(this->mem_case_->has_device_cuda_mem())
        << "DeviceType: " << device_tag
        << " not match MemoryCase: " << this->mem_case_->device_cuda_mem().DebugString();
    CHECK_OR_RETURN(
        parallel_desc.Containing(machine_id, this->mem_case_->device_cuda_mem().device_id()));
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

}  // namespace eager
}  // namespace oneflow
