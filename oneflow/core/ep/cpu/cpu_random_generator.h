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
#ifndef ONEFLOW_CORE_EP_CPU_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_EP_CPU_RANDOM_GENERATOR_H_

#include <array>
#include <cmath>
#include <math.h>
#include <random>
#include <mutex>

#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/include/random_generator.h"

namespace oneflow {
namespace ep {

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

class CPUGenerator : public RandomGenerator {
 public:
  explicit CPUGenerator(uint64_t seed, int device_index)
      : RandomGenerator(), seed_(seed), engine_(seed), torch_engine_(seed) {}

  virtual ~CPUGenerator() = default;

  uint64_t current_seed() const override { return seed_; }
  void set_current_seed(uint64_t seed) override;

  std::mt19937& engine() { return engine_; }

  pytorch_mt19937_engine& torch_engine() { return torch_engine_; }

  std::string device_type_name() const override { return "cpu"; }
  int64_t device_index() const override { return 0; }

  size_t GetStateSize() const override;
  void GetState(size_t state_size, void* state) const override;
  void SetState(size_t state_size, const void* state) override;

 public:
  mutable std::mutex mutex_;
  uint64_t seed_;
  std::mt19937 engine_;
  // TODO(Liang Depeng): needed to implement the get_state/set_state of pytorch_mt_19937_engine
  //                     refer to
  //                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/CPUGenerator.cpp#L206
  pytorch_mt19937_engine torch_engine_;
};

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_RANDOM_GENERATOR_H_
