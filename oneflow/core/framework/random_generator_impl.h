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

#include <array>
#include <cmath>
#include <cstdint>
#include <math.h>
#include <mutex>
#include <random>
#include <unordered_map>

#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/device.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif  // WITH_CUDA

namespace oneflow {

#ifdef WITH_CUDA

namespace ep {

class CudaStream;

}  // namespace ep

#endif  // WITH_CUDA

namespace one {

class Tensor;

class GeneratorImpl;

namespace detail {

template<typename T>
Maybe<T> MakeGeneratorImpl(uint64_t seed, int device_index);

Maybe<GeneratorImpl> MakeGeneratorImpl(uint64_t seed, DeviceType device_type, int device_index);

struct DeviceKey {
  DeviceType device_type = kInvalidDevice;
  int device_index = -1;
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

  virtual Maybe<Symbol<Device>> device() const = 0;

  virtual Maybe<Tensor> GetState() const = 0;
  virtual Maybe<void> SetState(const std::shared_ptr<Tensor>& tensor_state) = 0;

 protected:
  uint64_t seed_;
};

class DeviceGeneratorImpl : public GeneratorImpl {
 public:
  explicit DeviceGeneratorImpl(const uint64_t& seed, DeviceType device_type, int device_index)
      : GeneratorImpl(seed), device_type_(device_type), device_index_(device_index) {}

  virtual ~DeviceGeneratorImpl() = default;

  const DeviceType& device_type() const { return device_type_; }
  int device_index() const { return device_index_; }

  std::mutex mutex_;

 protected:
  DeviceType device_type_;
  int device_index_;
};

// NOTE(Liang Depeng): The following implementation of mt19937 is modified from
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/MT19937RNGEngine.h
//                     in order to make distribution related cpu kernels to have the same output as
//                     pytorch when setting the same seed.
constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

struct pytorch_mt19937_data_pod {
  uint64_t seed_;
  int left_;
  bool seeded_;
  uint32_t next_;
  std::array<uint32_t, MERSENNE_STATE_N> state_;
};

class pytorch_mt19937_engine {
 public:
  inline explicit pytorch_mt19937_engine(uint64_t seed = 5489) { init_with_uint32(seed); }

  inline pytorch_mt19937_data_pod data() const { return data_; }

  inline void set_data(pytorch_mt19937_data_pod data) { data_ = data; }

  inline uint64_t seed() const { return data_.seed_; }

  inline bool is_valid() {
    if ((data_.seeded_ == true) && (data_.left_ > 0 && data_.left_ <= MERSENNE_STATE_N)
        && (data_.next_ <= MERSENNE_STATE_N)) {
      return true;
    }
    return false;
  }

  inline uint32_t operator()() {
    uint32_t y;

    if (--(data_.left_) == 0) { next_state(); }
    y = *(data_.state_.data() + data_.next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
  }

 private:
  pytorch_mt19937_data_pod data_;

  inline void init_with_uint32(uint64_t seed) {
    data_.seed_ = seed;
    data_.seeded_ = true;
    data_.state_[0] = seed & 0xffffffff;
    for (int j = 1; j < MERSENNE_STATE_N; ++j) {
      data_.state_[j] = (1812433253 * (data_.state_[j - 1] ^ (data_.state_[j - 1] >> 30)) + j);
    }
    data_.left_ = 1;
    data_.next_ = 0;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) { return (u & UMASK) | (v & LMASK); }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  inline void next_state() {
    uint32_t* p = data_.state_.data();
    data_.left_ = MERSENNE_STATE_N;
    data_.next_ = 0;

    for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for (int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
  }
};

class CPUGeneratorImpl : public DeviceGeneratorImpl {
 public:
  explicit CPUGeneratorImpl(uint64_t seed)
      : DeviceGeneratorImpl(seed, DeviceType::kCPU, 0), engine_(seed), torch_engine_(seed) {}

  virtual ~CPUGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override;

  std::mt19937& engine() { return engine_; }

  pytorch_mt19937_engine& torch_engine() { return torch_engine_; }

  Maybe<Symbol<Device>> device() const override { return Device::New("cpu", device_index()); }

  Maybe<Tensor> GetState() const override;
  Maybe<void> SetState(const std::shared_ptr<Tensor>& tensor_state) override;

 public:
  std::mt19937 engine_;
  // TODO(Liang Depeng): needed to implement the get_state/set_state of pytorch_mt_19937_engine
  //                     refer to
  //                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/CPUGeneratorImpl.cpp#L206
  pytorch_mt19937_engine torch_engine_;
};

#ifdef WITH_CUDA

class CUDAGeneratorImpl : public DeviceGeneratorImpl {
 public:
  explicit CUDAGeneratorImpl(uint64_t seed, int device_index);
  virtual ~CUDAGeneratorImpl() = default;

  int32_t max_block_num() const { return max_block_num_; }
  int32_t max_thread_num() const { return max_thread_num_; }

  void set_current_seed(uint64_t seed) override;

  Maybe<Symbol<Device>> device() const override { return Device::New("cuda", device_index()); }

  Maybe<Tensor> GetState() const override;
  Maybe<void> SetState(const std::shared_ptr<Tensor>& tensor_state) override;
  std::tuple<uint64_t, dim3, dim3> CalcExecutionPolicy(int64_t total_elements,
                                                       ep::CudaStream* stream);

  uint64_t get_philox_offset(uint64_t increment);

 private:
  int32_t max_block_num_;
  int32_t max_thread_num_;
  uint64_t philox_offset_per_thread_ = 0;
};

#endif  // WITH_CUDA

class AutoGeneratorImpl : public GeneratorImpl {
 public:
  AutoGeneratorImpl(uint64_t seed) : GeneratorImpl(seed) {}
  virtual ~AutoGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override;

  Maybe<Symbol<Device>> device() const override { return Device::New("auto"); }

  Maybe<Tensor> GetState() const override;
  Maybe<void> SetState(const std::shared_ptr<Tensor>& tensor_state) override;

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
