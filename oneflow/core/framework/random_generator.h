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
#include <fcntl.h>
#include <unistd.h>

#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

class GeneratorImpl {
 public:
  GeneratorImpl() = default;
  GeneratorImpl(const uint64_t& seed, const std::string& device_type)
      : seed_(seed), device_type_(device_type) {}

  virtual ~GeneratorImpl() = default;

  virtual void set_current_seed(uint64_t seed) = 0;
  uint64_t current_seed() const { return seed_; }

  const std::string& device_type() const { return device_type_; }

 protected:
  uint64_t seed_;
  std::string device_type_;
};

template<DeviceType device_type>
class DeviceGeneratorImpl;

template<>
class DeviceGeneratorImpl<DeviceType::kCPU> : public GeneratorImpl {
 public:
  DeviceGeneratorImpl(uint64_t seed) : GeneratorImpl(seed, "cpu"), mt19937_generator_(seed) {}

  virtual ~DeviceGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    mt19937_generator_.seed(seed_);
  }

  std::mt19937& generator() { return mt19937_generator_; }

 public:
  std::mt19937 mt19937_generator_;
};

#ifdef WITH_CUDA
template<>
class DeviceGeneratorImpl<DeviceType::kGPU> : public GeneratorImpl {
 public:
  DeviceGeneratorImpl(uint64_t seed);

  virtual ~DeviceGeneratorImpl();

  const int32_t& block_num() const { return block_num_; }
  const int32_t& thread_num() const { return thread_num_; }

  curandState* curand_states() const { return curand_states_; }

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    CudaRandInit(seed_);
  }

 private:
  void CudaRandInit(uint64_t seed);

  int32_t block_num_;
  int32_t thread_num_;
  curandState* curand_states_;
};
#endif  // WITH_CUDA

class AutoGeneratorImpl : public GeneratorImpl {
 public:
  AutoGeneratorImpl(uint64_t seed) : GeneratorImpl(seed, "auto") {}

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    for (const auto& it : generators_) { it.second->set_current_seed(seed); }
  }

  template<DeviceType device_type>
  Maybe<DeviceGeneratorImpl<device_type>> GetDeviceGenerator() {
    CHECK_OR_RETURN(device_type != DeviceType::kInvalidDevice);
    auto it = generators_.find(device_type);
    if (it == generators_.end()) {
      it = generators_
               .emplace(device_type, std::make_shared<DeviceGeneratorImpl<device_type>>(seed_))
               .first;
    }
    return std::dynamic_pointer_cast<DeviceGeneratorImpl<device_type>>(it->second);
  }

 private:
  std::map<DeviceType, std::shared_ptr<GeneratorImpl>> generators_;
};

class Generator final {
 public:
  // The default seed is selected to be a large number
  // with good distribution of 0s and 1s in bit representation
  static constexpr uint64_t default_rng_seed_val = 67280421310721;

 public:
  Generator() = default;

  Maybe<void> Init(const std::string& device, uint64_t seed);

  static Maybe<Generator> New(const std::string& device) {
    return New(device, default_rng_seed_val);
  }
  static Maybe<Generator> New(const std::string& device, uint64_t seed);

  void set_current_seed(uint64_t seed) { gen_impl_->set_current_seed(seed); }

  uint64_t current_seed() const { return gen_impl_->current_seed(); }

  // Reset current seed by the default seed, and returns it.
  uint64_t seed();

 private:
  std::shared_ptr<GeneratorImpl> gen_impl_;
};

void ManualSeed(uint64_t seed);

const std::shared_ptr<AutoGeneratorImpl>& GetDefaultAutoGenerator();

std::shared_ptr<AutoGeneratorImpl> CreateAutoGenerator(uint64_t seed);

template<DeviceType device_type>
std::shared_ptr<DeviceGeneratorImpl<device_type>> CreateDeviceGenerator(uint64_t seed);

template<DeviceType device_type>
const std::shared_ptr<DeviceGeneratorImpl<device_type>>& GetDefaultDeviceGenerator();

template<DeviceType device_type>
Maybe<DeviceGeneratorImpl<device_type>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImpl>& generator);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
