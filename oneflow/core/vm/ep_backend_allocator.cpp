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
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {
namespace vm {

Maybe<void> EpBackendAllocator::Allocate(char** mem_ptr, std::size_t size) {
  return ep_device_->Alloc(allocation_options_, reinterpret_cast<void**>(mem_ptr), size);
}

void EpBackendAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  ep_device_->Free(allocation_options_, mem_ptr);
}

void EpBackendAllocator::DeviceReset() {
#ifdef WITH_CUDA
  if (ep_device_->device_type() == DeviceType::kCUDA) {
    ep_device_->SetAsActiveDevice();
    // NOTE(chengcheng): In some corner case on ubuntu, cuda memory not released even if OOM.
    //   So there need release all cuda memory allocated by this process before core dump.
    LOG(WARNING) << "OOM error is detected, process will exit. And it will start to reset CUDA "
                 << "device for releasing device memory.";
    OF_CUDA_CHECK(cudaDeviceReset());
  }
#endif
}

}  // namespace vm
}  // namespace oneflow
