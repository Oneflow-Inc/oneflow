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
#endif

namespace oneflow {
namespace one {

uint64_t getNonDeterministicRandom();

void manual_seed(uint64_t seed);

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation
constexpr uint64_t default_rng_seed_val = 67280421310721;

template<DeviceType device_type>
class GeneratorImpl;

class GeneratorImplBase {
 public:
  GeneratorImplBase() = default;
  virtual ~GeneratorImplBase() = default;

  virtual void set_seed(const uint64_t seed) = 0;
  virtual uint64_t get_seed() const { return seed_; }
  virtual DeviceType device_type() const { return device_type_; }

 protected:
  DeviceType device_type_;
  uint64_t seed_;
};

template<>
class GeneratorImpl<DeviceType::kAUTO> : public GeneratorImplBase {
 public:
  GeneratorImpl(uint64_t seed) {
    seed_ = seed;
    device_type_ = DeviceType::kAUTO;
  }

  void set_seed(const uint64_t seed) override {
    for (const auto& it : generators_) { it.second->set_seed(seed); }
  }

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

 private:
  std::map<DeviceType, std::shared_ptr<GeneratorImplBase>> generators_;
};

template<>
class GeneratorImpl<DeviceType::kCPU> : public GeneratorImplBase {
 public:
  GeneratorImpl(uint64_t seed) : mt19937_generator_(seed) {
    seed_ = seed;
    device_type_ = DeviceType::kCPU;
  }
  virtual ~GeneratorImpl() = default;

  void set_seed(const uint64_t seed) override {
    seed_ = seed;
    mt19937_generator_.seed(seed_);
  }

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
  void CudaRandInit(uint64_t seed);

  void set_seed(const uint64_t seed) override {
    seed_ = seed;
    CudaRandInit(seed_);
  }

 protected:
  curandState* curand_states_;
  int32_t block_num_;
  int32_t thread_num_;
};
#endif

class Generator final {
 public:
  explicit Generator(std::string device = "auto", uint64_t seed = default_rng_seed_val) {
    if (device == "cpu") {
      gen_impl_ = std::make_shared<GeneratorImpl<DeviceType::kCPU>>(seed);
    } else if (device == "cuda") {
      gen_impl_ = std::make_shared<GeneratorImpl<DeviceType::kGPU>>(seed);
    } else if (device == "auto") {
      gen_impl_ = std::make_shared<GeneratorImpl<DeviceType::kAUTO>>(seed);
    } else {
      UNIMPLEMENTED() << " device unimplemented, device name: " << device;
    }
  }

  void set_seed(const uint64_t seed) { gen_impl_->set_seed(seed); }

  uint64_t get_seed() const { return gen_impl_->get_seed(); }

  uint64_t seed() {
    uint64_t seed = getNonDeterministicRandom();
    set_seed(seed);
    return seed;
  }

 private:
  std::shared_ptr<GeneratorImplBase> gen_impl_;
};

template<DeviceType device_type>
const std::shared_ptr<GeneratorImpl<device_type>> CreateGenerator(uint64_t seed) {
  return std::make_shared<GeneratorImpl<device_type>>(seed);
}

template<DeviceType device_type>
const std::shared_ptr<GeneratorImpl<device_type>>& GetDefaultGenerator() {
  static auto generator = CreateGenerator<device_type>(getNonDeterministicRandom());
  return generator;
}

template<DeviceType device_type>
const Maybe<GeneratorImpl<device_type>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImplBase>& generator) {
  const auto& gen = std::dynamic_pointer_cast<GeneratorImpl<device_type>>(generator);
  if (gen->device_type() == DeviceType::kAUTO) {
    return gen->template GetDeviceGenerator<device_type>();
  }
  CHECK_OR_RETURN(gen->device_type() == device_type);
  return gen;
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
