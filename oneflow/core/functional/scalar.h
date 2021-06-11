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

#ifndef ONEFLOW_CORE_FUNCTIONAL_SCALAR_H_
#define ONEFLOW_CORE_FUNCTIONAL_SCALAR_H_

#include <type_traits>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/functional/value_types.h"

namespace oneflow {
namespace one {
namespace functional {

class Scalar {
 public:
  Scalar() : Scalar(int32_t(0)) {}

  template<typename T, typename std::enable_if<
                           std::is_integral<T>::value && std::is_signed<T>::value, int>::type = 0>
  explicit Scalar(const T& value)
      : value_type_(ValueTypeOf<T>()), value_{.s = value}, active_tag_(HAS_S) {}

  template<typename T, typename std::enable_if<
                           std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type = 0>
  explicit Scalar(const T& value)
      : value_type_(ValueTypeOf<T>()), value_{.u = value}, active_tag_(HAS_U) {}

  template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
  explicit Scalar(const T& value)
      : value_type_(ValueTypeOf<T>()), value_{.d = value}, active_tag_(HAS_D) {}

  const ValueType& type() const { return value_type_; }

  template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
  Maybe<T> As() const {
    switch (active_tag_) {
      case HAS_S: return static_cast<T>(value_.s);
      case HAS_U: return static_cast<T>(value_.u);
      case HAS_D: return static_cast<T>(value_.d);
      default: UNIMPLEMENTED_THEN_RETURN() << "The scalar has not been initialized.";
    }
  }

  bool IsIntegral() const { return active_tag_ == HAS_S || active_tag_ == HAS_U; }
  bool IsFloatingPoint() const { return active_tag_ == HAS_D; }
  bool IsSigned() const { return active_tag_ == HAS_S || active_tag_ == HAS_D; }
  bool IsUnsigned() const { return active_tag_ == HAS_U; }

 private:
  ValueType value_type_;
  union Value {
    int64_t s;
    uint64_t u;
    double d;
  } value_;
  enum { HAS_S, HAS_U, HAS_D, HAS_NONE } active_tag_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_SCALAR_H_
