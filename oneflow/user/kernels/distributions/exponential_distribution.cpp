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
#include <math.h>
#include <array>
#include <cmath>
#include <cstdint>

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/exponential_distribution.h"

namespace oneflow {

// NOTE(Liang Depeng): The following implementation of mt19937 is copyed from
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/MT19937RNGEngine.h
//                     in order to align with pytorch.
constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

struct mt19937_data_pod {
  uint64_t seed_;
  int left_;
  bool seeded_;
  uint32_t next_;
  std::array<uint32_t, MERSENNE_STATE_N> state_;
};

class mt19937_engine {
public:

  inline explicit mt19937_engine(uint64_t seed = 5489) {
    init_with_uint32(seed);
  }

  inline mt19937_data_pod data() const {
    return data_;
  }

  inline void set_data(mt19937_data_pod data) {
    data_ = data;
  }

  inline uint64_t seed() const {
    return data_.seed_;
  }

  inline bool is_valid() {
    if ((data_.seeded_ == true)
      && (data_.left_ > 0 && data_.left_ <= MERSENNE_STATE_N)
      && (data_.next_ <= MERSENNE_STATE_N)) {
      return true;
    }
    return false;
  }

  inline uint32_t operator()() {
    uint32_t y;

    if (--(data_.left_) == 0) {
        next_state();
    }
    y = *(data_.state_.data() + data_.next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
  }

private:
  mt19937_data_pod data_;

  inline void init_with_uint32(uint64_t seed) {
    data_.seed_ = seed;
    data_.seeded_ = true;
    data_.state_[0] = seed & 0xffffffff;
    for (int j = 1; j < MERSENNE_STATE_N; ++j) {
      data_.state_[j] = (1812433253 * (data_.state_[j-1] ^ (data_.state_[j-1] >> 30)) + j);
    }
    data_.left_ = 1;
    data_.next_ = 0;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  inline void next_state() {
    uint32_t* p = data_.state_.data();
    data_.left_ = MERSENNE_STATE_N;
    data_.next_ = 0;

    for(int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for(int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
  }

};

typedef mt19937_engine mt19937;

inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

template <typename T, typename V>
inline T uniform_real(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  T x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

template<typename T, typename E = void>
class CPUExponentialDistributionImpl;

template<typename T>
class CPUExponentialDistributionImpl<T,
                                 typename std::enable_if<std::is_floating_point<T>::value>::type> {
 public:
  CPUExponentialDistributionImpl() {}

  T operator()(mt19937& engine, const T lambd) { 
    uint32_t random1 = engine();
    uint32_t random2 = engine();
    uint64_t rand_unit = make64BitsFrom32Bits(random1, random2);
    T random_val = uniform_real(rand_unit, 0.0, 1.0);
    return static_cast<T>(-1.0) / lambd * std::log(static_cast<T>(1.0) - random_val); 
  }
};

template<typename T>
void ExponentialDistribution<DeviceType::kCPU, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  mt19937 engine{gen->current_seed()};
  CPUExponentialDistributionImpl<T> impl;
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = impl(engine, lambd_); }
}

#define INITIATE_CPU_UNIFORM_DISTRIBUTION(T, typeproto)               \
  template void ExponentialDistribution<DeviceType::kCPU, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,            \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
