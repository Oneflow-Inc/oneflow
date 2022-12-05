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
#ifndef ONEFLOW_CORE_COMMON_MATH_UTIL_H_
#define ONEFLOW_CORE_COMMON_MATH_UTIL_H_
#include <stdint.h>
#include "data_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

int64_t Gcd(int64_t m, int64_t n);

int64_t Lcm(int64_t m, int64_t n);

template<typename T>
OF_DEVICE_FUNC T DeviceMin(T a, T b) {
#if defined(__CUDA_ARCH__)
  return a < b ? a : b;
#else
  return std::min(a, b);
#endif
}

template<typename T>
OF_DEVICE_FUNC T DeviceMax(T a, T b) {
#if defined(__CUDA_ARCH__)
  return a > b ? a : b;
#else
  return std::max(a, b);
#endif
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_MATH_UTIL_H_
