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
#ifndef ONEFLOW_CORE_COMMON_STATIC_CHECK_H_
#define ONEFLOW_CORE_COMMON_STATIC_CHECK_H_

#include "oneflow/core/common/type_traits.h"

namespace oneflow {

namespace static_check {

template<typename... Args>
void ForEachArgsType(Args... args);

template<template<typename> class Checker>
inline void ForEachArgsType() {}

template<template<typename> class Checker, typename T, typename... Args>
void ForEachArgsType(T a, Args... args) {
  Checker<T> check{};
  ForEachArgsType<Checker>(args...);
};

template<typename T>
struct CheckNotOutArg {
  static_assert(!(std::is_pointer<T>::value && !std::is_const<T>::value), "");
  static_assert(!(std::is_reference<T>::value && !std::is_const<T>::value), "");
};

template<typename T>
struct CheckIsScalarType {
  static_assert(IsScalarType<typename std::decay<T>::type>::value, "");
};

}  // namespace static_check

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STATIC_CHECK_H_
