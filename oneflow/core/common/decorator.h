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
#ifndef ONEFLOW_CORE_COMMON_DECORATOR_H_
#define ONEFLOW_CORE_COMMON_DECORATOR_H_

#include "oneflow/core/common/tuple_hash.h"
#include "oneflow/core/common/static_check.h"

namespace oneflow {

template<template<typename...> class Decorator>
struct WithDecorator final {
  template<typename T, typename = void>
  struct Decorate;
  template<typename T, typename... Args>
  struct Decorate<T (*)(Args...)> final {
    template<T (*func)(Args...)>
    static T Call(Args... args) {
      return Decorator<T, Args...>::template Call<func>(args...);
    }
  };
};

#define DECORATED(decorator, fn_ptr) \
  &WithDecorator<decorator>::Decorate<decltype(fn_ptr)>::Call<fn_ptr>

#define THREAD_LOCAL_STATIC_FUNC_TEMPLATE(T, Args)                           \
  template<T (*func)(Args...)>                                               \
  static T Call(Args... args) {                                              \
    using KeyT = std::tuple<typename std::decay<Args>::type...>;             \
    static thread_local std::unordered_map<KeyT, T> map;                     \
    const auto& key = KeyT(args...);                                         \
    auto iter = map.find(key);                                               \
    if (iter == map.end()) { iter = map.emplace(key, func(args...)).first; } \
    return iter->second;                                                     \
  }

// for scalar type key.
template<typename T, typename... Args>
struct ThreadLocal final {
  THREAD_LOCAL_STATIC_FUNC_TEMPLATE(T, Args);

 private:
  static void StaticCheck() {
    auto* _0 = &static_check::ForEachArgsType<static_check::CheckIsScalarType, Args...>;
    auto* _1 = &static_check::ForEachArgsType<static_check::CheckNotOutArg, Args...>;
  }
};

// for deep copiable type key.
template<typename T, typename... Args>
struct ThreadLocalDeepCopiable final {
  THREAD_LOCAL_STATIC_FUNC_TEMPLATE(T, Args);

 private:
  static void StaticCheck() {
    auto* _ = &static_check::ForEachArgsType<static_check::CheckNotOutArg, Args...>;
  }
};

#undef THREAD_LOCAL_STATIC_FUNC_TEMPLATE

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DECORATOR_H_
