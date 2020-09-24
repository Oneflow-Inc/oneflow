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

void CudaHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  OF_CUDA_CHECK(cudaMallocHost(mem_ptr, size));
}

void CudaHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  OF_CUDA_CHECK(cudaFreeHost(mem_ptr));
}

}  // namespace vm
}  // namespace oneflow

#endif
