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
#ifndef ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

template<DeviceType device_type>
class RandomGenerator;

template<>
class RandomGenerator<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator(int64_t seed, ep::Stream* stream) : mt19937_generator_(seed) {}
  ~RandomGenerator() {}

  template<typename T>
  void Uniform(const int64_t elem_cnt, T* dptr);
  template<typename T>
  void Uniform(const int64_t elem_cnt, const T min, const T max, T* dptr);

 private:
  std::mt19937 mt19937_generator_;
};

template<>
class RandomGenerator<DeviceType::kCUDA> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator(int64_t seed, ep::Stream* stream);
  ~RandomGenerator();

  template<typename T>
  void Uniform(const int64_t elem_cnt, T* dptr);

 private:
#ifdef WITH_CUDA
  curandGenerator_t curand_generator_;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
