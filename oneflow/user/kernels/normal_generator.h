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
#ifndef ONEFLOW_USER_KERNELS_NORMAL_GENERATOR_H_
#define ONEFLOW_USER_KERNELS_NORMAL_GENERATOR_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/random_generator.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {

template<DeviceType device_type>
class NormalGenerator;

template<>
class NormalGenerator<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalGenerator);
  NormalGenerator(const std::shared_ptr<one::Generator>& generator) {
    generator_ = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  }
  ~NormalGenerator() = default;

  template<typename T>
  void Generate(DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr);

 private:
  std::shared_ptr<one::CPUGeneratorImpl> generator_;
};

#ifdef WITH_CUDA
template<>
class NormalGenerator<DeviceType::kGPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalGenerator);
  NormalGenerator(const std::shared_ptr<one::Generator>& generator) {
    generator_ = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
  }
  ~NormalGenerator() = default;

  template<typename T>
  void Generate(DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr);

 private:
  std::shared_ptr<one::CUDAGeneratorImpl> generator_;
};
#endif

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_NORMAL_GENERATOR_H_