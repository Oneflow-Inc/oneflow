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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTION_TRAITS_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTION_TRAITS_H_

#include <tuple>

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template<typename T, typename R, typename... Args>
struct function_traits<R (T::*)(Args...)> : public function_traits<R(Args...)> {};

template<typename T, typename R, typename... Args>
struct function_traits<R (T::*)(Args...) const> : public function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<R (*)(Args...)> : public function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<R(Args...)> {
  using func_type = R(Args...);
  using return_type = R;
  using argument_types = std::tuple<Args...>;
  static constexpr size_t number_of_arguments = sizeof...(Args);
};

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTION_TRAITS_H_
