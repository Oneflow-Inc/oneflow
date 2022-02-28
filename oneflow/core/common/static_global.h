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
#ifndef ONEFLOW_CORE_COMMON_STATIC_GLOBAL_H_
#define ONEFLOW_CORE_COMMON_STATIC_GLOBAL_H_

#include <mutex>
#include "oneflow/core/common/decorator.h"

namespace oneflow {

template<typename... Args>
struct StaticGlobalCopiable;

template<typename RetT>
struct StaticGlobalCopiable<RetT> {
  template<RetT (*func)()>
  static RetT Call() {
    static RetT value = func();
    return value;
  }
};

template<typename RetT, typename Arg0>
struct StaticGlobalCopiable<RetT, Arg0> {
  template<RetT (*func)(Arg0)>
  static RetT Call(Arg0 arg0) {
    using KeyT = typename std::decay<Arg0>::type;
    using MappedT = typename std::decay<RetT>::type;
    static std::mutex mutex;
    static std::unordered_map<KeyT, MappedT> map;
    {
      std::unique_lock<std::mutex> lock(mutex);
      auto iter = map.find(arg0);
      if (iter != map.end()) { return iter->second; }
    }
    auto obj = func(arg0);
    {
      std::unique_lock<std::mutex> lock(mutex);
      return map.emplace(arg0, std::move(obj)).first->second;
    }
  }

 private:
  static_assert(!IsOutArg<Arg0>::value, "");
  static_assert(!StaticAny<IsOutArg, Arg0>::value, "");
};

template<typename RetT, typename Arg0, typename Arg1, typename... Args>
struct StaticGlobalCopiable<RetT, Arg0, Arg1, Args...> {
  template<RetT (*func)(Arg0, Arg1, Args...)>
  static RetT Call(Arg0 arg0, Arg1 arg1, Args... args) {
    using KeyT0 = typename std::decay<Arg0>::type;
    using KeyT1 = typename std::decay<Arg1>::type;
    using KeyT = std::tuple<KeyT0, KeyT1, typename std::decay<Args>::type...>;
    using MappedT = typename std::decay<RetT>::type;
    static std::mutex mutex;
    static std::unordered_map<KeyT, MappedT> map;
    const auto& key = KeyT(arg0, arg1, args...);
    {
      std::unique_lock<std::mutex> lock(mutex);
      auto iter = map.find(key);
      if (iter != map.end()) { return iter->second; }
    }
    auto obj = func(arg0, arg1, args...);
    {
      std::unique_lock<std::mutex> lock(mutex);
      return map.emplace(key, std::move(obj)).first->second;
    }
  }

 private:
  static_assert(!StaticAny<IsOutArg, Arg0, Arg1, Args...>::value, "");
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STATIC_GLOBAL_H_
