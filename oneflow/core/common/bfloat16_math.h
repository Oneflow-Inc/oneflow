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
#ifndef ONEFLOW_CORE_COMMON_BFLOAT16_MATH_H_
#define ONEFLOW_CORE_COMMON_BFLOAT16_MATH_H_

#include "oneflow/core/common/bfloat16.h"

namespace std {

// reference: pytorch/c10/util/BFloat16-math.h
// https://github.com/pytorch/pytorch/blob/release/1.12/c10/util/BFloat16-math.h
inline oneflow::bfloat16 acos(oneflow::bfloat16 a) { return std::acos(static_cast<float>(a)); }
inline oneflow::bfloat16 asin(oneflow::bfloat16 a) { return std::asin(static_cast<float>(a)); }
inline oneflow::bfloat16 atan(oneflow::bfloat16 a) { return std::atan(static_cast<float>(a)); }
inline oneflow::bfloat16 erf(oneflow::bfloat16 a) { return std::erf(static_cast<float>(a)); }
inline oneflow::bfloat16 erfc(oneflow::bfloat16 a) { return std::erfc(static_cast<float>(a)); }
inline oneflow::bfloat16 exp(oneflow::bfloat16 a) { return std::exp(static_cast<float>(a)); }
inline oneflow::bfloat16 expm1(oneflow::bfloat16 a) { return std::expm1(static_cast<float>(a)); }
inline oneflow::bfloat16 log(oneflow::bfloat16 a) { return std::log(static_cast<float>(a)); }
inline oneflow::bfloat16 log10(oneflow::bfloat16 a) { return std::log10(static_cast<float>(a)); }
inline oneflow::bfloat16 log1p(oneflow::bfloat16 a) { return std::log1p(static_cast<float>(a)); }
inline oneflow::bfloat16 log2(oneflow::bfloat16 a) { return std::log2(static_cast<float>(a)); }
inline oneflow::bfloat16 ceil(oneflow::bfloat16 a) { return std::ceil(static_cast<float>(a)); }
inline oneflow::bfloat16 cos(oneflow::bfloat16 a) { return std::cos(static_cast<float>(a)); }
inline oneflow::bfloat16 floor(oneflow::bfloat16 a) { return std::floor(static_cast<float>(a)); }
inline oneflow::bfloat16 nearbyint(oneflow::bfloat16 a) {
  return std::nearbyint(static_cast<float>(a));
}
inline oneflow::bfloat16 sin(oneflow::bfloat16 a) { return std::sin(static_cast<float>(a)); }
inline oneflow::bfloat16 tan(oneflow::bfloat16 a) { return std::tan(static_cast<float>(a)); }
inline oneflow::bfloat16 sinh(oneflow::bfloat16 a) { return std::sinh(static_cast<float>(a)); }
inline oneflow::bfloat16 cosh(oneflow::bfloat16 a) { return std::cosh(static_cast<float>(a)); }
inline oneflow::bfloat16 tanh(oneflow::bfloat16 a) { return std::tanh(static_cast<float>(a)); }
inline oneflow::bfloat16 trunc(oneflow::bfloat16 a) { return std::trunc(static_cast<float>(a)); }
inline oneflow::bfloat16 lgamma(oneflow::bfloat16 a) { return std::lgamma(static_cast<float>(a)); }
inline oneflow::bfloat16 sqrt(oneflow::bfloat16 a) { return std::sqrt(static_cast<float>(a)); }
inline oneflow::bfloat16 rsqrt(oneflow::bfloat16 a) {
  return 1.0 / std::sqrt(static_cast<float>(a));
}
inline oneflow::bfloat16 abs(oneflow::bfloat16 a) { return std::abs(static_cast<float>(a)); }
inline oneflow::bfloat16 pow(oneflow::bfloat16 a, double b) {
  return std::pow(static_cast<float>(a), b);
}
inline oneflow::bfloat16 pow(oneflow::bfloat16 a, oneflow::bfloat16 b) {
  return std::pow(static_cast<float>(a), static_cast<float>(b));
}
inline oneflow::bfloat16 fmod(oneflow::bfloat16 a, oneflow::bfloat16 b) {
  return std::fmod(static_cast<float>(a), static_cast<float>(b));
}

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_BFLOAT16_MATH_H_
