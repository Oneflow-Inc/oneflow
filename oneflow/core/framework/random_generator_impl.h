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

#include <mutex>
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

class GeneratorImpl;

namespace detail {

template<DeviceType device_type>
Maybe<GeneratorImpl> MakeGeneratorImpl(uint64_t seed, int device_index);

struct DeviceKey {
  DeviceType device_type;
  int device_index;
};

bool operator==(const DeviceKey& k1, const DeviceKey& k2);

struct DeviceKeyHash {
  size_t operator()(const DeviceKey& key) const;
};

template<DeviceType device_type>
DeviceKey GetCurrentDeviceKey();

}  // namespace detail

class GeneratorImpl {
 public:
  GeneratorImpl() = default;
  explicit GeneratorImpl(const uint64_t& seed, const detail::DeviceKey& device_key)
      : seed_(seed), device_key_(device_key) {}

  virtual ~GeneratorImpl() = default;

  virtual void set_current_seed(uint64_t seed) = 0;
  uint64_t current_seed() const { return seed_; }

  int device_index() const { return device_key_.device_index; }
  const DeviceType& device_type() const { return device_key_.device_type; }
  const detail::DeviceKey& device_key() const { return device_key_; }

 protected:
  uint64_t seed_;
  detail::DeviceKey device_key_;
};

class CPUGeneratorImpl : public GeneratorImpl {
 public:
  explicit CPUGeneratorImpl(uint64_t seed)
      : GeneratorImpl(seed, detail::DeviceKey{DeviceType::kCPU, 0}), engine_(seed) {}

  virtual ~CPUGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    engine_.seed(seed_);
  }

  std::mt19937& engine() { return engine_; }
  void set_engine(std::mt19937 engine) { engine_ = engine; }

 public:
  std::mt19937 engine_;
};

#ifdef WITH_CUDA
class CUDAGeneratorImpl : public GeneratorImpl {
 public:
  explicit CUDAGeneratorImpl(uint64_t seed, int device_index);
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

namespace detail {

int GetCudaDeviceCount();

void InitCurandStates(uint64_t seed, int32_t block_num, int32_t thread_num, curandState* states);

}  // namespace detail
#endif  // WITH_CUDA

class AutoGeneratorImpl {
 public:
  AutoGeneratorImpl(uint64_t seed) : seed_(seed), enable_auto_create_(true) {}

  AutoGeneratorImpl(const std::shared_ptr<GeneratorImpl>& impl)
      : seed_(impl->current_seed()), enable_auto_create_(false) {
    generators_.emplace(impl->device_key(), impl);
  }
  virtual ~AutoGeneratorImpl() = default;

  uint64_t current_seed() const { return seed_; }

  void set_current_seed(uint64_t seed) {
    seed_ = seed;
    for (const auto& it : generators_) { it.second->set_current_seed(seed); }
  }

  template<DeviceType device_type>
  Maybe<GeneratorImpl> GetOrCreateDeviceGenerator(int device_index) {
    CHECK_OR_RETURN(device_type != DeviceType::kInvalidDevice);
    detail::DeviceKey device_key;
    if (device_index == -1) {
      device_key = detail::GetCurrentDeviceKey<device_type>();
    } else {
      device_key = detail::DeviceKey{device_type, device_index};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = generators_.find(device_key);
    if (it == generators_.end()) {
      CHECK_OR_RETURN(enable_auto_create_)
          << "There is no generator for device " << device_type << ".";
      it =
          generators_
              .emplace(device_key,
                       JUST(detail::MakeGeneratorImpl<device_type>(seed_, device_key.device_index)))
              .first;
    }
    return it->second;
  }

 private:
  mutable std::mutex mutex_;
  uint64_t seed_;
  bool enable_auto_create_;
  std::unordered_map<detail::DeviceKey, std::shared_ptr<GeneratorImpl>, detail::DeviceKeyHash>
      generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_IMPL_H_
