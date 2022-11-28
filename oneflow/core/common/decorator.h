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

#include <type_traits>
#include <unordered_map>
#include "tuple_hash.h"
#include "static_check.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/cpp_attribute.h"

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

#define DECORATE(fn_ptr, decorator) \
  (&WithDecorator<decorator>::Decorate<decltype(fn_ptr)>::Call<fn_ptr>)

template<typename... Args>
struct ThreadLocalCopiable;

template<typename RetT>
struct ThreadLocalCopiable<RetT> {
  template<RetT (*func)()>
  static RetT Call() {
    static thread_local RetT value = func();
    return value;
  }
};

template<typename RetT, typename Arg0>
struct ThreadLocalCopiable<RetT, Arg0> {
  template<RetT (*func)(Arg0)>
  static RetT Call(Arg0 arg0) {
    using KeyT = typename std::decay<Arg0>::type;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<KeyT, MappedT> map;
    auto iter = map.find(arg0);
    if (iter == map.end()) { iter = map.emplace(arg0, func(arg0)).first; }
    return iter->second;
  }

 private:
  static_assert(!IsOutArg<Arg0>::value, "");
  static_assert(!StaticAny<IsOutArg, Arg0>::value, "");
};

template<typename RetT, typename Arg0, typename Arg1>
struct ThreadLocalCopiable<RetT, Arg0, Arg1> {
  template<RetT (*func)(Arg0, Arg1)>
  static RetT Call(Arg0 arg0, Arg1 arg1) {
    using KeyT0 = typename std::decay<Arg0>::type;
    using KeyT1 = typename std::decay<Arg1>::type;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<KeyT0, std::unordered_map<KeyT1, MappedT>> map;
    auto* last_map = &map[arg0];
    auto iter = last_map->find(arg1);
    if (iter == last_map->end()) { iter = last_map->emplace(arg1, func(arg0, arg1)).first; }
    return iter->second;
  }

 private:
  static_assert(!StaticAny<IsOutArg, Arg0, Arg1>::value, "");
};

template<typename RetT, typename Arg0, typename Arg1, typename Arg2>
struct ThreadLocalCopiable<RetT, Arg0, Arg1, Arg2> {
  template<RetT (*func)(Arg0, Arg1, Arg2)>
  static RetT Call(Arg0 arg0, Arg1 arg1, Arg2 arg2) {
    using KeyT0 = typename std::decay<Arg0>::type;
    using KeyT1 = typename std::decay<Arg1>::type;
    using KeyT2 = typename std::decay<Arg2>::type;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<
        KeyT0, std::unordered_map<KeyT1, std::unordered_map<KeyT2, MappedT>>>
        map;
    auto* last_map = &map[arg0][arg1];
    auto iter = last_map->find(arg2);
    if (iter == last_map->end()) { iter = last_map->emplace(arg2, func(arg0, arg1, arg2)).first; }
    return iter->second;
  }

 private:
  static_assert(!StaticAny<IsOutArg, Arg0, Arg1, Arg2>::value, "");
};

template<typename RetT, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
         typename... Args>
struct ThreadLocalCopiable<RetT, Arg0, Arg1, Arg2, Arg3, Args...> {
  template<RetT (*func)(Arg0, Arg1, Arg2, Arg3, Args...)>
  static RetT Call(Arg0 arg0, Arg1 arg1, Arg2 arg2, Arg3 arg3, Args... args) {
    using KeyT0 = typename std::decay<Arg0>::type;
    using KeyT1 = typename std::decay<Arg1>::type;
    using KeyT2 = typename std::decay<Arg2>::type;
    using KeyT3 = typename std::decay<Arg3>::type;
    using KeyT = std::tuple<KeyT0, KeyT1, KeyT2, KeyT3, typename std::decay<Args>::type...>;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<KeyT, MappedT> map;
    const auto& key = KeyT(arg0, arg1, arg2, arg3, args...);
    auto iter = map.find(key);
    if (iter == map.end()) { iter = map.emplace(key, func(arg0, arg1, arg2, arg3, args...)).first; }
    return iter->second;
  }

 private:
  static_assert(!StaticAny<IsOutArg, Arg0, Arg1, Arg2, Arg3, Args...>::value, "");
};

// for scalar type key.
template<typename RetT, typename... Args>
struct ThreadLocal : public ThreadLocalCopiable<RetT, Args...> {
 private:
  static_assert(StaticAll<IsDecayedScalarType, Args...>::value, "");
};

template<typename... Args>
struct ThreadLocalCachedCopiable;

template<typename RetT>
struct ThreadLocalCachedCopiable<RetT> {
  template<RetT (*func)()>
  static RetT Call() {
    static thread_local RetT value = func();
    return value;
  }
};

template<typename RetT, typename Arg0>
struct ThreadLocalCachedCopiable<RetT, Arg0> {
  template<RetT (*func)(Arg0)>
  static RetT Call(Arg0 arg0) {
    using KeyT = typename std::decay<Arg0>::type;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<KeyT, MappedT> map;
    auto iter = map.find(arg0);
    if (iter == map.end()) {
      if (unlikely(map.size() >= ThreadLocalEnvInteger<ONEFLOW_THRAED_LOCAL_CACHED_SIZE>())) {
        map.clear();
      }
      iter = map.emplace(arg0, func(arg0)).first;
    }
    return iter->second;
  }

 private:
  static_assert(!IsOutArg<Arg0>::value, "");
  static_assert(!StaticAny<IsOutArg, Arg0>::value, "");
};

template<typename RetT, typename Arg0, typename... Args>
struct ThreadLocalCachedCopiable<RetT, Arg0, Args...> {
  template<RetT (*func)(Arg0, Args...)>
  static RetT Call(Arg0 arg0, Args... args) {
    using KeyT0 = typename std::decay<Arg0>::type;
    using KeyT = std::tuple<KeyT0, typename std::decay<Args>::type...>;
    using MappedT = typename std::decay<RetT>::type;
    static thread_local std::unordered_map<KeyT, MappedT> map;
    const auto& key = KeyT(arg0, args...);
    auto iter = map.find(key);
    if (iter == map.end()) {
      if (unlikely(map.size() >= ThreadLocalEnvInteger<ONEFLOW_THRAED_LOCAL_CACHED_SIZE>())) {
        map.clear();
      }
      iter = map.emplace(key, func(arg0, args...)).first;
    }
    return iter->second;
  }

 private:
  static_assert(!StaticAny<IsOutArg, Arg0, Args...>::value, "");
};

// for scalar type key.
template<typename RetT, typename... Args>
struct ThreadLocalCached : public ThreadLocalCachedCopiable<RetT, Args...> {
 private:
  static_assert(StaticAll<IsDecayedScalarType, Args...>::value, "");
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DECORATOR_H_
