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
#ifdef WITH_NPU

#include "oneflow/core/vm/npu_host_allocator.h"
#include "oneflow/core/device/npu_util.h"

namespace oneflow {
namespace vm {

NpuHostAllocator::~NpuHostAllocator() {
  std::cout<<"~NpuHostAllocator()"<<device_id_<<std::endl;
  NpuCurrentDeviceGuard guard(device_id_);
  for (const auto& ptr_vec : granularity2free_ptrs_) {
    for (char* ptr : ptr_vec) { OF_NPU_CHECK(aclrtFreeHost(ptr)); }
  }
  for (const auto& pair : occupied_ptr2granularity_) { OF_NPU_CHECK(aclrtFreeHost(pair.first)); }
}

void NpuHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  std::size_t granularity = std::ceil(std::log2(size));
  CHECK_GE(granularity, 0);
  CHECK_LT(granularity, kMaxGranularity);
  CHECK_LE(size, 1 << granularity);
  NpuCurrentDeviceGuard guard(device_id_);
  std::unique_lock<std::mutex> lock(mutex_);
  auto* vec = &granularity2free_ptrs_[granularity];
  if (vec->empty()) {
    void* ptr = nullptr;
    OF_NPU_CHECK(aclrtMallocHost(&ptr, 1 << granularity));
    vec->emplace_back((char*)ptr);
  }
  *mem_ptr = vec->back();
  vec->pop_back();
  occupied_ptr2granularity_[*mem_ptr] = granularity;
}

void NpuHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = occupied_ptr2granularity_.find(mem_ptr);
  CHECK(iter != occupied_ptr2granularity_.end());
  std::size_t granularity = iter->second;
  occupied_ptr2granularity_.erase(iter);
  granularity2free_ptrs_[granularity].emplace_back(mem_ptr);
}

}  // namespace vm
}  // namespace oneflow

#endif
