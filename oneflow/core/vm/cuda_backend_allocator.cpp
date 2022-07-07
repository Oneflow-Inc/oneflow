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

#include "oneflow/core/vm/cuda_backend_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include <iostream>

namespace oneflow {
namespace vm {

Maybe<void> CudaBackendAllocator::Allocate(char** mem_ptr, std::size_t size) {
  cudaSetDevice(device_id_);
  if (cudaMalloc(mem_ptr, size) != cudaSuccess) {
    *mem_ptr = nullptr;
    return Error::OutOfMemoryError() << "cuda allocator out of memory";
  }
  return Maybe<void>::Ok();
}

void CudaBackendAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  cudaSetDevice(device_id_);
  OF_CUDA_CHECK(cudaFree(mem_ptr));
}

void CudaBackendAllocator::DeviceReset() {
  cudaSetDevice(device_id_);
  // NOTE(chengcheng): In some corner case on ubuntu, cuda memory not released even if OOM.
  //   So there need release all cuda memory allocated by this process before core dump.
  LOG(WARNING) << "OOM error is detected, process will exit. And it will start to reset CUDA "
               << "device for releasing device memory.";
  OF_CUDA_CHECK(cudaDeviceReset());
}

}  // namespace vm
}  // namespace oneflow

#endif
