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

#ifndef ONEFLOW_CORE_COMMON_SCALAR_H_
#define ONEFLOW_CORE_COMMON_SCALAR_H_

#include <type_traits>
#include <complex>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class Scalar {
 public:
  Scalar() : Scalar(int32_t(0)) {}

  template<typename T, typename std::enable_if<std::is_same<std::complex<float>, T>::value
                                                   || std::is_same<std::complex<double>, T>::value,
                                               int>::type = 0>
  Scalar(const T& value) : value_{.c = {value.real(), value.imag()}}, active_tag_(HAS_C) {}

  template<typename T, typename std::enable_if<std::is_same<T, bool>::value, int>::type = 0>
  OF_DEVICE_FUNC Scalar(const T& value) : value_{.b = value}, active_tag_(HAS_B) {}

  template<typename T, typename std::enable_if<
                           std::is_integral<T>::value && std::is_signed<T>::value, int>::type = 0>
  OF_DEVICE_FUNC Scalar(const T& value) : value_{.s = value}, active_tag_(HAS_S) {}

  template<typename T,
           typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value
                                       && !std::is_same<T, bool>::value,
                                   int>::type = 0>
  OF_DEVICE_FUNC Scalar(const T& value) : value_{.u = value}, active_tag_(HAS_U) {}

  template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
  OF_DEVICE_FUNC Scalar(const T& value) : value_{.d = value}, active_tag_(HAS_D) {}

  template<typename T, typename std::enable_if<!std::is_same<T, Scalar>::value, int>::type = 0>
  OF_DEVICE_FUNC Scalar& operator=(const T& value) {
    *this = Scalar(value);
    return *this;
  }

  OF_DEVICE_FUNC Scalar& operator=(const Scalar& other) {
    value_ = other.value_;
    active_tag_ = other.active_tag_;
    return *this;
  }

  template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
  OF_DEVICE_FUNC T As() const {
    switch (active_tag_) {
      case HAS_B: return static_cast<T>(value_.b);
      case HAS_S: return static_cast<T>(value_.s);
      case HAS_U: return static_cast<T>(value_.u);
      case HAS_D: return static_cast<T>(value_.d);
      default: assert(false); return 0;
    }
  }

  template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
  OF_DEVICE_FUNC T Value() const {
    return As<T>();
  }

  template<typename T, typename std::enable_if<std::is_same<std::complex<float>, T>::value
                                                   || std::is_same<std::complex<double>, T>::value,
                                               int>::type = 0>
  T Value() const {
    if (!IsComplex()) { return T(As<double>(), 0.0); }
    return T(value_.c.real, value_.c.imag);
  }

  bool IsBool() const { return active_tag_ == HAS_B; }
  bool IsIntegral() const { return active_tag_ == HAS_S || active_tag_ == HAS_U; }
  bool IsFloatingPoint() const { return active_tag_ == HAS_D; }
  bool IsSigned() const { return active_tag_ == HAS_S || active_tag_ == HAS_D; }
  bool IsUnsigned() const { return active_tag_ == HAS_U; }
  bool IsComplex() const { return active_tag_ == HAS_C; }

  Scalar operator+(const Scalar& other) const;
  Scalar operator-(const Scalar& other) const;
  Scalar operator*(const Scalar& other) const;
  Scalar operator/(const Scalar& other) const;

  Scalar& operator+=(const Scalar& other);
  Scalar& operator-=(const Scalar& other);
  Scalar& operator*=(const Scalar& other);
  Scalar& operator/=(const Scalar& other);

 private:
  union Value {
    bool b;
    int64_t s;
    uint64_t u;
    double d;
    struct {
      double real;
      double imag;
    } c;
  } value_;
  enum { HAS_B, HAS_S, HAS_U, HAS_D, HAS_C, HAS_NONE } active_tag_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SCALAR_H_
