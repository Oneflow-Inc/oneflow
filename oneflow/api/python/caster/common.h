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
#ifndef ONEFLOW_API_PYTHON_CASTER_COMMON_H_
#define ONEFLOW_API_PYTHON_CASTER_COMMON_H_

#include <type_traits>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

// The condition follows the pybind11 source code
template<typename T>
using IsSupportedByPybind11WhenInsideSharedPtr =
    std::is_base_of<type_caster_base<T>, type_caster<T>>;

#define PYBIND11_TYPE_CASTER_WITH_SHARED_PTR(type, py_name)                               \
 protected:                                                                               \
  std::shared_ptr<type> value;                                                            \
                                                                                          \
 public:                                                                                  \
  static constexpr auto name = py_name;                                                   \
  template<typename T_, enable_if_t<std::is_same<type, remove_cv_t<T_>>::value, int> = 0> \
  static handle cast(T_* src, return_value_policy policy, handle parent) {                \
    if (!src) return none().release();                                                    \
    if (policy == return_value_policy::take_ownership) {                                  \
      auto h = cast(std::move(*src), policy, parent);                                     \
      delete src;                                                                         \
      return h;                                                                           \
    }                                                                                     \
    return cast(*src, policy, parent);                                                    \
  }                                                                                       \
  operator type*() { return value.get(); }                                                \
  operator type&() { return *value; }                                                     \
  operator type&&()&& { return std::move(*value); }                                       \
  template<typename T_>                                                                   \
  using cast_op_type = pybind11::detail::movable_cast_op_type<T_>

}  // namespace detail
}  // namespace pybind11

#endif  // ONEFLOW_API_PYTHON_CASTER_COMMON_H_
