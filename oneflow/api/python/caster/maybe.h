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
#ifndef ONEFLOW_API_PYTHON_CASTER_MAYBE_H_
#define ONEFLOW_API_PYTHON_CASTER_MAYBE_H_
#include <pybind11/pybind11.h>

#include "oneflow/api/python/caster/common.h"
#include "oneflow/core/common/maybe.h"

namespace pybind11 {
namespace detail {

using oneflow::Maybe;

namespace impl {

template<typename T>
using IsHoldedInsideSharedPtrByMaybe =
    std::is_same<decltype(
                     std::declval<Maybe<T>>().Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()),
                 std::shared_ptr<T>>;

template<typename T, typename std::enable_if_t<IsSupportedByPybind11WhenInsideSharedPtr<T>::value
                                                   && IsHoldedInsideSharedPtrByMaybe<T>::value,
                                               int> = 0>
std::shared_ptr<T> GetOrThrowHelper(Maybe<T> x) {
  return x.GetPtrOrThrow();
}

template<typename T, typename std::enable_if_t<!IsSupportedByPybind11WhenInsideSharedPtr<T>::value
                                                   || !IsHoldedInsideSharedPtrByMaybe<T>::value,
                                               int> = 0>
T GetOrThrowHelper(Maybe<T> x) {
  return x.GetOrThrow();
}

}  // namespace impl

// Information about pybind11 custom type caster can be found
// at oneflow/api/python/caster/optional.h, and also at
// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
template<typename Type>
struct maybe_caster {
  using Value = decltype(impl::GetOrThrowHelper(std::declval<Type>()));
  using value_conv = make_caster<Value>;

  bool load(handle src, bool convert) {
    if (!src) { return false; }
    if (src.is_none()) {
      // Maybe<T> (except Maybe<void>) does not accept `None` from Python. Users can use Optional in
      // those cases.
      return false;
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) { return false; }

    value = std::make_shared<Type>(cast_op<Value&&>(std::move(inner_caster)));
    return true;
  }

  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<Value>::policy(policy);
    }
    return value_conv::cast(impl::GetOrThrowHelper(std::forward<T>(src)), policy, parent);
  }

  PYBIND11_TYPE_CASTER_WITH_SHARED_PTR(Maybe<void>, _("Maybe[void]"));
};

template<>
struct maybe_caster<Maybe<void>> {
  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src.IsOk()) { oneflow::ThrowError(src.stacked_error()); }
    return none().inc_ref();
  }

  bool load(handle src, bool convert) {
    if (src && src.is_none()) {
      return true;  // None is accepted because NoneType (i.e. void) is the value type of
                    // Maybe<void>
    }
    return false;
  }

  PYBIND11_TYPE_CASTER_WITH_SHARED_PTR(Maybe<void>, _("Maybe[void]"));
};

template<typename T>
struct type_caster<Maybe<T>> : public maybe_caster<Maybe<T>> {};

}  // namespace detail
}  // namespace pybind11

#endif  // ONEFLOW_API_PYTHON_CASTER_MAYBE_H_
