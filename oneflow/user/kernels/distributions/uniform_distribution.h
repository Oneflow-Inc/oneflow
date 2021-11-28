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
#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_DISTRIBUTION_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_DISTRIBUTION_H_

#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/framework/random_generator.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {

template<DeviceType device_type, typename T>
class UniformDistribution;

template<typename T>
class UniformDistribution<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniformDistribution);
  UniformDistribution(T low, T high) : low_(low), high_(high) {}
  ~UniformDistribution() = default;

  void operator()(ep::Stream* stream, const int64_t elem_cnt, T* dptr,
                  const std::shared_ptr<one::Generator>& generator) const;

 private:
  const T low_;
  const T high_;
};

#ifdef WITH_CUDA
template<typename T>
class UniformDistribution<DeviceType::kCUDA, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniformDistribution);
  UniformDistribution(T low, T high) : low_(low), high_(high) {}
  ~UniformDistribution() = default;

  void operator()(ep::Stream* stream, const int64_t elem_cnt, T* dptr,
                  const std::shared_ptr<one::Generator>& generator) const;

 private:
  const T low_;
  const T high_;
};
#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_DISTRIBUTION_H_
