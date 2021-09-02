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

#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/maybe.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

class GeneratorImpl;

namespace detail {

template<typename T>
Maybe<T> MakeGeneratorImpl(uint64_t seed, int device_index);

struct DeviceKey {
  DeviceType device_type;
  int device_index;
};

struct DeviceKeyHash {
  size_t operator()(const DeviceKey& key) const;
};

bool operator==(const DeviceKey& k1, const DeviceKey& k2);

template<typename T>
DeviceKey MakeDeviceKey(int device_index);

}  // namespace detail

class GeneratorImpl {
 public:
  explicit GeneratorImpl(const uint64_t& seed) : seed_(seed) {}

  virtual ~GeneratorImpl() = default;

  virtual void set_current_seed(uint64_t seed) = 0;
  uint64_t current_seed() const { return seed_; }

 protected:
  uint64_t seed_;
};

class DeviceGeneratorImpl : public GeneratorImpl {
 public:
  explicit DeviceGeneratorImpl(const uint64_t& seed, const detail::DeviceKey& device_key)
      : GeneratorImpl(seed), device_key_(device_key) {}

  virtual ~DeviceGeneratorImpl() = default;

  int device_index() const { return device_key_.device_index; }
  const DeviceType& device_type() const { return device_key_.device_type; }
  const detail::DeviceKey& device_key() const { return device_key_; }

 protected:
  detail::DeviceKey device_key_;
};

class CPUGeneratorImpl : public DeviceGeneratorImpl {
 public:
  explicit CPUGeneratorImpl(uint64_t seed)
      : DeviceGeneratorImpl(seed, detail::DeviceKey{DeviceType::kCPU, 0}), engine_(seed) {}

  virtual ~CPUGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    engine_.seed(seed_);
  }

  std::mt19937& engine() { return engine_; }

 public:
  std::mt19937 engine_;
};

#ifdef WITH_CUDA
class CUDAGeneratorImpl : public DeviceGeneratorImpl {
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

class AutoGeneratorImpl : public GeneratorImpl {
 public:
  AutoGeneratorImpl(uint64_t seed) : GeneratorImpl(seed) {}
  virtual ~AutoGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override {
    std::lock_guard<std::mutex> lock(mutex_);
    seed_ = seed;
    for (const auto& it : generators_) { it.second->set_current_seed(seed); }
  }

  template<typename T>
  Maybe<T> GetOrCreate(int device_index) {
    detail::DeviceKey device_key = detail::MakeDeviceKey<T>(device_index);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = generators_.find(device_key);
    if (it == generators_.end()) {
      it = generators_
               .emplace(device_key,
                        JUST(detail::MakeGeneratorImpl<T>(seed_, device_key.device_index)))
               .first;
    }
    auto impl = std::dynamic_pointer_cast<T>(it->second);
    CHECK_NOTNULL_OR_RETURN(impl);
    return impl;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<detail::DeviceKey, std::shared_ptr<GeneratorImpl>, detail::DeviceKeyHash>
      generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_IMPL_H_
