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
#ifndef ONEFLOW_CORE_EP_CUDA_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_EP_CUDA_RANDOM_GENERATOR_H_

#ifdef WITH_CUDA

#include <mutex>
#include <curand.h>
#include <curand_kernel.h>

#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/random_generator.h"

namespace oneflow {
namespace ep {

class CUDAGenerator : public RandomGenerator {
 public:
  explicit CUDAGenerator(uint64_t seed, int device_index);
  virtual ~CUDAGenerator() = default;

  int32_t max_block_num() const { return max_block_num_; }
  int32_t max_thread_num() const { return max_thread_num_; }

  uint64_t current_seed() const override { return seed_; }
  void set_current_seed(uint64_t seed) override;

  std::string device_type_name() const override { return "cuda"; }
  int64_t device_index() const override { return device_index_; }

  size_t GetStateSize() const override;
  void GetState(size_t state_size, void* state) const override;
  void SetState(size_t state_size, const void* state) override;

  std::tuple<uint64_t, dim3, dim3> CalcExecutionPolicy(int64_t total_elements, CudaStream* stream);

  uint64_t get_philox_offset(uint64_t increment);

 public:
  mutable std::mutex mutex_;

 private:
  uint64_t seed_;
  int64_t device_index_;
  int32_t max_block_num_;
  int32_t max_thread_num_;
  uint64_t philox_offset_per_thread_ = 0;
};

}  // namespace ep
}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_RANDOM_GENERATOR_H_
