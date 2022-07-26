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
#ifndef ONEFLOW_CORE_COMMON_FUNCTOR_TRAITS_H_
#define ONEFLOW_CORE_COMMON_FUNCTOR_TRAITS_H_

namespace oneflow {

template<typename X, typename = void>
struct Functor4FuncPtr;
template<typename T, typename... Args>
struct Functor4FuncPtr<T (*)(Args...)> final {
  template<T (*func)(Args...)>
  struct Functor final {
    T operator()(Args... args) { return func(args...); }
  };
};

template<typename T, typename... Args>
struct Functor4FuncPtr<T(Args...)> final {
  template<T(func)(Args...)>
  struct Functor final {
    T operator()(Args... args) { return func(args...); }
  };
};

#define FUNCTOR_CLASS_FOR_FUNC_PTR(ptr) ::oneflow::Functor4FuncPtr<decltype(ptr)>::Functor<ptr>

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

#define FUNC_PTR_FOR_FUNCTOR_CLASS(cls) \
  (&::oneflow::FuncPtrForFunctor<decltype(&cls::operator())>::Call)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FUNCTOR_TRAITS_H_
