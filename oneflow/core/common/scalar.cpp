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

#include <complex>
#include "oneflow/core/common/scalar.h"

namespace oneflow {

#define DEFINE_SCALAR_BINARY_OP(op)                                             \
  Scalar& Scalar::operator op##=(const Scalar& other) {                         \
    if (IsComplex() || other.IsComplex()) {                                     \
      std::complex<double> val =                                                \
          Value<std::complex<double>>() op other.Value<std::complex<double>>(); \
      *this = val;                                                              \
    }                                                                           \
    if (IsFloatingPoint() || other.IsFloatingPoint()) {                         \
      double val = As<double>() op other.As<double>();                          \
      *this = val;                                                              \
    } else {                                                                    \
      int64_t val = As<int64_t>() op other.As<int64_t>();                       \
      *this = val;                                                              \
    }                                                                           \
    return *this;                                                               \
  }                                                                             \
  Scalar Scalar::operator op(const Scalar& other) const {                       \
    if (IsComplex() || other.IsComplex()) {                                     \
      std::complex<double> val =                                                \
          Value<std::complex<double>>() op other.Value<std::complex<double>>(); \
      return Scalar(val);                                                       \
    }                                                                           \
    if (IsFloatingPoint() || other.IsFloatingPoint()) {                         \
      double val = As<double>() op other.As<double>();                          \
      return Scalar(val);                                                       \
    }                                                                           \
    int64_t val = As<int64_t>() op other.As<int64_t>();                         \
    return Scalar(val);                                                         \
  }

DEFINE_SCALAR_BINARY_OP(+);
DEFINE_SCALAR_BINARY_OP(-);
DEFINE_SCALAR_BINARY_OP(*);
DEFINE_SCALAR_BINARY_OP(/);  // NOLINT
#undef DEFINE_SCALAR_BINARY_OP

}  // namespace oneflow
