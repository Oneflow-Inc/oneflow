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
#ifndef ONEFLOW_CORE_COMMON_FUNCTOR_UTIL_H_
#define ONEFLOW_CORE_COMMON_FUNCTOR_UTIL_H_

#include "oneflow/core/common/decorator.h"

namespace oneflow {

template<typename X, typename = void>
struct FuncPtrForFunctor;

template<typename Functor, typename RetT, typename... Args>
struct FuncPtrForFunctor<RetT (Functor::*)(Args...)> final {
  static RetT Call(Args... args) { return Functor()(args...); }
};

template<typename Functor, typename RetT, typename... Args>
struct FuncPtrForFunctor<RetT (Functor::*)(Args...) const> final {
  static RetT Call(Args... args) { return Functor()(args...); }
};

#define FUNC_PTR_FOR_FUNCTOR(cls) (&::oneflow::FuncPtrForFunctor<decltype(&cls::operator())>::Call)

#define CACHED_FUNCTOR_PTR(cls) DECORATE(FUNC_PTR_FOR_FUNCTOR(cls), ThreadLocalCachedCopiable)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FUNCTOR_UTIL_H_
