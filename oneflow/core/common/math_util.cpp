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
#include <utility>
#include "glog/logging.h"
#include "oneflow/core/common/math_util.h"

namespace oneflow {

int64_t Gcd(int64_t m, int64_t n) {
  if (m < n) { std::swap(m, n); }
  if (n == 0) { return m; }
  CHECK_GT(m, 0);
  CHECK_GT(n, 0);
  return Gcd(n, m % n);
}

int64_t Lcm(int64_t m, int64_t n) { return m * n / Gcd(m, n); }

template<typename T>
T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) { result = result * x + A[i]; }
  return result;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 */

double calc_digamma_cpu(double x) {
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = std::modf(x, &q);
    return calc_digamma_cpu(1 - x) - pi<double> / tan(pi<double> * r);
  }

  // Push x to be >= 10
  double result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) { return result + PSI_10; }

  // Compute asymptotic digamma
  static const double A[] = {
      8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
      -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y = z * polevl(z, A, 6);
  }
  return result + log(x) - (0.5 / x) - y;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 */

float calc_digamma_cpu(float x) {
  static float PSI_10 = 2.25175258906672110764f;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == truncf(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<float>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = std::modf(x, &q);
    float pi_over_tan_pi_x = (float)(pi<double> / tan(pi<double> * r));
    return calc_digamma_cpu(1 - x) - pi_over_tan_pi_x;
  }

  // Push x to be >= 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) { return result + PSI_10; }

  // Compute asymptotic digamma
  static const float A[] = {
      8.33333333333333333333E-2f,  -2.10927960927960927961E-2f, 7.57575757575757575758E-3f,
      -4.16666666666666666667E-3f, 3.96825396825396825397E-3f,  -8.33333333333333333333E-3f,
      8.33333333333333333333E-2f,
  };

  float y = 0;
  if (x < 1.0e17f) {
    float z = 1 / (x * x);
    y = z * polevl(z, A, 6);
  }
  return result + logf(x) - (0.5f / x) - y;
}

}  // namespace oneflow
