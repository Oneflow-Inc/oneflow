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
#ifndef ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_IMPL_H_
#define ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_IMPL_H_

#include <random>
#include <unordered_map>

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/maybe.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

class GeneratorImpl {
 public:
  GeneratorImpl() = default;
  explicit GeneratorImpl(const uint64_t& seed, const DeviceType& device_type)
      : seed_(seed), device_type_(device_type) {}

  virtual ~GeneratorImpl() = default;

  virtual void set_current_seed(uint64_t seed) = 0;
  uint64_t current_seed() const { return seed_; }

  const DeviceType& device_type() const { return device_type_; }

 protected:
  uint64_t seed_;
  DeviceType device_type_;
};

class CPUGeneratorImpl : public GeneratorImpl {
 public:
  explicit CPUGeneratorImpl(uint64_t seed)
      : GeneratorImpl(seed, DeviceType::kCPU), mt19937_generator_(seed) {}

  virtual ~CPUGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    mt19937_generator_.seed(seed_);
  }

  std::mt19937& generator() { return mt19937_generator_; }

 public:
  std::mt19937 mt19937_generator_;
};

#ifdef WITH_CUDA
class CUDAGeneratorImpl : public GeneratorImpl {
 public:
  explicit CUDAGeneratorImpl(uint64_t seed);
  virtual ~CUDAGeneratorImpl();

  int32_t max_block_num() const { return max_block_num_; }
  int32_t max_thread_num() const { return max_thread_num_; }

  curandState* curand_states() const { return curand_states_; }

  void set_current_seed(uint64_t seed) override;

 private:
  int32_t max_block_num_;
  int32_t max_thread_num_;
  curandState* curand_states_;
};

void InitCurandStates(uint64_t seed, int32_t block_num, int32_t thread_num,
                      curandState* curand_states);
#endif  // WITH_CUDA

class AutoGeneratorImpl {
 public:
  AutoGeneratorImpl(uint64_t seed) : seed_(seed), enable_auto_create_(true) {}

  AutoGeneratorImpl(const std::shared_ptr<GeneratorImpl>& impl)
      : seed_(impl->current_seed()), enable_auto_create_(false) {
    generators_.emplace(impl->device_type(), impl);
  }

  virtual ~AutoGeneratorImpl() = default;

  uint64_t current_seed() const { return seed_; }

  void set_current_seed(uint64_t seed) {
    seed_ = seed;
    for (const auto& it : generators_) { it.second->set_current_seed(seed); }
  }

  template<DeviceType device_type>
  Maybe<GeneratorImpl> GetOrCreateDeviceGenerator() {
    CHECK_OR_RETURN(device_type != DeviceType::kInvalidDevice);
    auto it = generators_.find(device_type);
    if (it == generators_.end()) {
      CHECK_OR_RETURN(enable_auto_create_)
          << "There is no generator for device " << device_type << ".";
      it = generators_.emplace(device_type, JUST(MakeGeneratorImpl<device_type>(seed_))).first;
    }
    return it->second;
  }

  template<DeviceType device_type>
  Maybe<GeneratorImpl> MakeGeneratorImpl(uint64_t seed);

 private:
  uint64_t seed_;
  bool enable_auto_create_;
  std::unordered_map<DeviceType, std::shared_ptr<GeneratorImpl>, std::hash<int>> generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_IMPL_H_
