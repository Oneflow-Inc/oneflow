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
#ifndef ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_

#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace vm {

static constexpr int kCudaHostMaxGranularity = 64;

class CudaHostAllocator final : public Allocator {
 public:
  CudaHostAllocator(const CudaHostAllocator&) = delete;
  CudaHostAllocator(CudaHostAllocator&&) = delete;
  CudaHostAllocator& operator=(const CudaHostAllocator&) = delete;
  CudaHostAllocator& operator=(CudaHostAllocator&&) = delete;

  explicit CudaHostAllocator(int64_t device_id) : Allocator(), device_id_(device_id) {}
  ~CudaHostAllocator() override;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void DeviceReset() override {}

 private:
  int64_t device_id_;
  std::mutex mutex_;
  std::array<std::vector<char*>, kCudaHostMaxGranularity> granularity2free_ptrs_;
  std::unordered_map<char*, size_t> occupied_ptr2granularity_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_
