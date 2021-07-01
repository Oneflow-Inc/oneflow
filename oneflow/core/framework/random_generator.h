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
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {
namespace one {

class GeneratorImplBase {
 public:
  GeneratorImplBase() = default;
  virtual ~GeneratorImplBase() = default;

 protected:
  DeviceType device_type_;
  uint64_t seed_;
};

template<DeviceType device_type>
class GeneratorImpl;

template<>
class GeneratorImpl<DeviceType::kCPU> : public GeneratorImplBase {
 public:
  GeneratorImpl(uint64_t seed) : mt19937_generator_(seed) {
    seed_ = seed;
    device_type_ = DeviceType::kCPU;
  }
  virtual ~GeneratorImpl() = default;

  std::mt19937& generator() { return mt19937_generator_; }

 public:
  std::mt19937 mt19937_generator_;
};

#ifdef WITH_CUDA
template<>
class GeneratorImpl<DeviceType::kGPU> : public GeneratorImplBase {
 public:
  GeneratorImpl(uint64_t seed);
  virtual ~GeneratorImpl();

  const int32_t& block_num() const { return block_num_; }
  const int32_t& thread_num() const { return thread_num_; }
  curandState* curand_states() const { return curand_states_; }

 protected:
  curandState* curand_states_;
  int32_t block_num_;
  int32_t thread_num_;
};
#endif

class Generator final {
 public:
  // TODO: make default value random like pytorh
  explicit Generator(uint64_t seed = 0) : seed_(seed) {}

  // TODO: should also set seed of generators?
  void set_current_seed(const uint64_t seed) { seed_ = seed; }
  uint64_t get_current_seed() const { return seed_; }

  template<DeviceType device_type>
  Maybe<GeneratorImpl<device_type>> GetDeviceGenerator() {
    CHECK_OR_RETURN(device_type != DeviceType::kInvalidDevice);
    auto it = generators_.find(device_type);
    if (it == generators_.end()) {
      it = generators_.emplace(device_type, std::make_shared<GeneratorImpl<device_type>>(seed_))
               .first;
    }
    return std::dynamic_pointer_cast<GeneratorImpl<device_type>>(it->second);
  }

  template<DeviceType device_type>
  static Maybe<GeneratorImpl<device_type>> GetDefaultDeviceGenerator() {
    CHECK_OR_RETURN(device_type != DeviceType::kInvalidDevice);
    const auto& generator = GetDefaultGenerator();
    return generator->GetDeviceGenerator<device_type>();
  }

  static const std::shared_ptr<Generator>& GetDefaultGenerator() {
    static auto generator = std::make_shared<Generator>();
    return generator;
  }

 private:
  uint64_t seed_;
  std::map<DeviceType, std::shared_ptr<GeneratorImplBase>> generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
