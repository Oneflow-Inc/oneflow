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
#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

TEST(CudaAllocator, cuda_allocator) {
  int gpu_num = -1;
  cudaGetDeviceCount(&gpu_num);
  if (gpu_num <= 0) {
    LOG(INFO) << "CudaAllocator Test: Skip because of non GPU device.";
    return;
  }
  ASSERT_TRUE(cudaSuccess == cudaSetDevice(0));
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  const size_t remain_bytes = 50 * 1048576;
  ASSERT_TRUE(cudaSuccess == cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes <= remain_bytes || free_bytes - remain_bytes < remain_bytes) {
    LOG(INFO) << "CudaAllocator Test: Skip because of allocator mem bytes less than 50MiB in GPU 0";
    return;
  }
  std::unique_ptr<Allocator> allo(new CudaAllocator(0));
  allo.reset(new SingleThreadOnlyAllocator(std::move(allo)));
  Allocator* a = allo.get();
  std::vector<char*> ptrs;
  for (int i = 0; i < 512; ++i) {
    char* ptr = nullptr;
    a->Allocate(&ptr, 1);
    ASSERT_TRUE(ptr != nullptr);
    ptrs.push_back(ptr);
  }
  std::sort(ptrs.begin(), ptrs.end());
  for (int i = 0; i < 512; ++i) {
    if (i > 0) {
      ASSERT_TRUE(ptrs.at(i) != ptrs.at(i - 1));
      ASSERT_TRUE(std::abs(ptrs.at(i) - ptrs.at(i - 1)) >= kCudaMemAllocAlignSize);
    }
    a->Deallocate(ptrs.at(i), 1);
  }

  ptrs.clear();
  for (int i = 0; i < 2048; ++i) {
    char* ptr = nullptr;
    a->Allocate(&ptr, 10000);
    ASSERT_TRUE(ptr != nullptr);
    ptrs.push_back(ptr);
  }
  std::sort(ptrs.begin(), ptrs.end());
  for (int i = 0; i < 2048; ++i) {
    if (i > 0) {
      ASSERT_TRUE(ptrs.at(i) != ptrs.at(i - 1));
      ASSERT_TRUE(std::abs(ptrs.at(i) - ptrs.at(i - 1)) >= kCudaMemAllocAlignSize);
    }
    a->Deallocate(ptrs.at(i), 10000);
  }

  char* data_ptr_1 = nullptr;
  a->Allocate(&data_ptr_1, 2048 * sizeof(float));

  char* data_ptr_2 = nullptr;
  a->Allocate(&data_ptr_2, 4096 * sizeof(double));

  ASSERT_TRUE(data_ptr_1 != data_ptr_2);
  if (data_ptr_1 < data_ptr_2) {
    ASSERT_TRUE(data_ptr_1 + 2048 * sizeof(float) <= data_ptr_2);
  } else {
    ASSERT_TRUE(data_ptr_2 + 4096 * sizeof(double) <= data_ptr_1);
  }

  a->Deallocate(data_ptr_2, 4096 * sizeof(double));
  a->Deallocate(data_ptr_1, 2048 * sizeof(float));
}

}  // namespace vm
}  // namespace oneflow

#endif  // WITH_CUDA
