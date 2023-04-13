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

static uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

template<typename T, typename V>
static T uniform_real(V val, T from, T to) {
  constexpr auto MASK =
      static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR =
      static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  T x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

template<typename T>
void ExponentialDistribution<DeviceType::kCPU, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  one::pytorch_mt19937_engine& engine = gen->torch_engine();
  for (int64_t i = 0; i < elem_cnt; ++i) {
    uint32_t random1 = engine();
    uint32_t random2 = engine();
    uint64_t rand_unit = make64BitsFrom32Bits(random1, random2);
    T random_val = uniform_real(rand_unit, 0.0, 1.0);
    dptr[i] = static_cast<T>(-1.0) / lambd_ * std::log(static_cast<T>(1.0) - random_val);
  }
}

#define INITIATE_CPU_UNIFORM_DISTRIBUTION(T, typeproto)                   \
  template void ExponentialDistribution<DeviceType::kCPU, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,                \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
