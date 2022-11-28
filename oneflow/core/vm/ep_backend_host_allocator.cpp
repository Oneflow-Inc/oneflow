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
#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

namespace vm {

Maybe<void> EpBackendHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  JUST(ep_device_->AllocPinned(allocation_options_, reinterpret_cast<void**>(mem_ptr), size));
  return Maybe<void>::Ok();
}

void EpBackendHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  ep_device_->FreePinned(allocation_options_, mem_ptr);
}

}  // namespace vm

}  // namespace oneflow
