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
#ifdef WITH_CUDA

#include "oneflow/core/vm/cuda_host_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

CudaHostAllocator::~CudaHostAllocator() {
  CudaCurrentDeviceGuard guard(device_id_);
  for (const auto& ptr_vec : granularity2free_ptrs_) {
    for (char* ptr : ptr_vec) { OF_CUDA_CHECK(cudaFreeHost(ptr)); }
  }
  for (const auto& pair : occupied_ptr2granularity_) { OF_CUDA_CHECK(cudaFreeHost(pair.first)); }
}

Maybe<void> CudaHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  std::size_t granularity = std::ceil(std::log2(size));
  CHECK_GE_OR_RETURN(granularity, 0) << "out of range";
  CHECK_LT_OR_RETURN(granularity, kCudaHostMaxGranularity) << "invalid granularity";
  CHECK_LE_OR_RETURN(size, 1 << granularity) << "out of range";
  CudaCurrentDeviceGuard guard(device_id_);
  std::unique_lock<std::mutex> lock(mutex_);
  auto* vec = &granularity2free_ptrs_[granularity];
  if (vec->empty()) {
    char* ptr = nullptr;
    OF_CUDA_CHECK(cudaMallocHost(&ptr, 1 << granularity));
    vec->emplace_back(ptr);
  }
  *mem_ptr = vec->back();
  vec->pop_back();
  occupied_ptr2granularity_[*mem_ptr] = granularity;
  return Maybe<void>::Ok();
}

void CudaHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = occupied_ptr2granularity_.find(mem_ptr);
  CHECK(iter != occupied_ptr2granularity_.end());
  std::size_t granularity = iter->second;
  occupied_ptr2granularity_.erase(iter);
  granularity2free_ptrs_[granularity].emplace_back(mem_ptr);
}

COMMAND(Singleton<CudaHostAllocator>::SetAllocated(new CudaHostAllocator(0)));

}  // namespace vm
}  // namespace oneflow

#endif
