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
#ifndef ONEFLOW_CORE_COMMON_FUNCTION_TRAITS_H_
#define ONEFLOW_CORE_COMMON_FUNCTION_TRAITS_H_

#include <tuple>
namespace oneflow {

template<typename... Args>
using void_t = void;

template<typename T, typename = void>
struct function_traits;

template<typename Ret, typename... Args>
struct function_traits<Ret (*)(Args...)> {
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
};

template<typename Ret, typename C, typename... Args>
struct function_traits<Ret (C::*)(Args...)> {
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
};

template<typename Ret, typename C, typename... Args>
struct function_traits<Ret (C::*)(Args...) const> {
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
};

template<typename F>
struct function_traits<F, void_t<decltype(&F::operator())>>
    : public function_traits<decltype(&F::operator())> {};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FUNCTION_TRAITS_H_
