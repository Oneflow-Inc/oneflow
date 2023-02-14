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
#ifndef ONEFLOW_API_PYTHON_CASTER_OPTIONAL_H_
#define ONEFLOW_API_PYTHON_CASTER_OPTIONAL_H_

#include <pybind11/pybind11.h>

#include "oneflow/api/python/caster/common.h"
#include "oneflow/core/common/optional.h"

namespace pybind11 {
namespace detail {

using oneflow::Optional;

namespace impl {

template<typename T>
T& DeferenceIfSharedPtr(std::shared_ptr<T> ptr) {
  return *ptr;
}

template<typename T>
T&& DeferenceIfSharedPtr(T&& obj) {
  return std::forward<T>(obj);
}

template<typename T>
using IsHoldedInsideSharedPtrByOptional =
    std::is_same<typename Optional<T>::storage_type, std::shared_ptr<T>>;

template<typename T, typename std::enable_if_t<IsSupportedByPybind11WhenInsideSharedPtr<T>::value
                                                   && IsHoldedInsideSharedPtrByOptional<T>::value,
                                               int> = 0>
std::shared_ptr<T> GetDataHelper(Optional<T> x) {
  return CHECK_JUST(x);
}

template<typename T, typename std::enable_if_t<!IsSupportedByPybind11WhenInsideSharedPtr<T>::value
                                                   || !IsHoldedInsideSharedPtrByOptional<T>::value,
                                               int> = 0>
T GetDataHelper(Optional<T> x) {
  return DeferenceIfSharedPtr<T>(CHECK_JUST(x));
}

}  // namespace impl

// Code is copied from pybind11 include/pybind11/stl.h
// Comments wrapped by /* */ are copied from
// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
template<typename Type>
struct oneflow_optional_caster {
  using Value = decltype(impl::GetDataHelper(std::declval<Type>()));
  using value_conv = make_caster<Value>;

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into a Optional<T>
   * instance or return false upon failure. The second argument
   * indicates whether implicit conversions should be applied.
   */
  bool load(handle src, bool convert) {
    if (!src) { return false; }
    if (src.is_none()) {
      return true;  // default-constructed value is already empty
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) { return false; }

    value = cast_op<Value&&>(std::move(inner_caster));
    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert an Optional<T> instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src) { return none().inc_ref(); }
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<Value>::policy(policy);
    }
    return value_conv::cast(impl::GetDataHelper(std::forward<T>(src)), policy, parent);
  }

  /**
   * This macro establishes the name 'Optional[T]' in
   * function signatures and declares a local variable
   * 'value' of type inty
   */
  PYBIND11_TYPE_CASTER(Type, _("Optional[") + value_conv::name + _("]"));
};

template<typename T>
struct type_caster<Optional<T>> : public oneflow_optional_caster<Optional<T>> {};

}  // namespace detail
}  // namespace pybind11

#endif  // ONEFLOW_API_PYTHON_CASTER_OPTIONAL_H_
