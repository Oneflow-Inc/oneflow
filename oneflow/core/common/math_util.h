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
 T polevl(const T x, const T A[], size_t len);

// This function references pytorch/aten/src/ATen/native/Math.h
double calc_digamma_cpu(double x);

float calc_digamma_cpu(float x);

template<typename scalar_t, typename accscalar_t>
static OF_DEVICE_FUNC scalar_t calc_digamma_cuda(scalar_t in) {
  static const double PI_f64 = 3.14159265358979323846;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  const accscalar_t A[] = {
      8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
      -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t x = static_cast<accscalar_t>(in);
  if (x == static_cast<accscalar_t>(0)) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(static_cast<scalar_t>(INFINITY), -x);
  }

  bool x_is_integer = x == trunc(x);
  accscalar_t result = static_cast<accscalar_t>(0);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return static_cast<scalar_t>(NAN);
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = modf(static_cast<double>(x), &q);
    result = static_cast<accscalar_t>(-PI_f64 / tan(PI_f64 * r));
    x = static_cast<accscalar_t>(1) - x;
  }

  while (x < 10) {
    result -= static_cast<accscalar_t>(1) / x;
    x += 1;
  }
  if (x == static_cast<accscalar_t>(10)) { return static_cast<scalar_t>(result + PSI_10); }

  accscalar_t y = 0;
  if (x < 1.0e17) {
    accscalar_t z = static_cast<accscalar_t>(1) / (x * x);

    accscalar_t polevl_result = 0;
    for (int i = 0; i <= 6; i++) { polevl_result = polevl_result * z + A[i]; }
    y = z * polevl_result;
  }

  return static_cast<scalar_t>(log(x) - (static_cast<accscalar_t>(0.5) / x) - y + result);
}

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

template<typename T>
constexpr T pi = static_cast<T>(3.141592653589793238462643383279502);

// template <typename T>
// inline constexpr T pi() {
//   return static_cast<T>(3.141592653589793238462643383279502);
// }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_MATH_UTIL_H_
