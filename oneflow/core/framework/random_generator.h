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
#ifndef ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/device_context.h"
// #ifdef WITH_CUDA
// #include <curand.h>
// #include <curand_kernel.h>
// #endif

namespace oneflow {
namespace one {

class GeneratorImplBase {
 public:
  GeneratorImplBase() = default;
  virtual ~GeneratorImplBase() = default;

 protected:
  int64_t seed_;
};

template<DeviceType device_type>
class GeneratorImpl;

template<>
class GeneratorImpl<DeviceType::kCPU> : public GeneratorImplBase {
 public:
  GeneratorImpl(int64_t seed) : mt19937_generator_(seed) {
    seed_ = seed;
    device_type_ = DeviceType::kCPU;
  }
  virtual ~GeneratorImpl() = default;

 protected:
  DeviceType device_type_;
  std::mt19937 mt19937_generator_;
};

// #ifdef WITH_CUDA
// template<>
// class GeneratorImpl<DeviceType::kGPU> : public GeneratorImplBase{
//  public:
//   GeneratorImpl(int64_t seed);
//   virtual ~GeneratorImpl() = default;

//  protected:
//   curandState* curand_states_;
//   int32_t block_num_;
//   int32_t thread_num_;
// };
// #endif

class Generator final {
 public:
  // TODO: make default value random like pytorh
  explicit Generator(int64_t seed = 0) : seed_(seed) {}

  void set_current_seed(const int64_t seed) { seed_ = seed; }

  template<DeviceType device_type>
  Maybe<std::shared_ptr<GeneratorImpl<device_type>>> GetDeviceGenerator() {
    CHECK_OR_RETURN(device_type == DeviceType::kInvalidDevice);
    auto it = generators_.find(device_type);
    if (it == generators_.end()) {
      it = generators_.emplace(device_type, std::make_shared<GeneratorImpl<device_type>>(seed_)).first;
    }
    return dynamic_cast<std::shared_ptr<GeneratorImpl<device_type>>>(it->second);
  }

 private:
  int64_t seed_;
  std::map<DeviceType, std::shared_ptr<GeneratorImplBase>> generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
