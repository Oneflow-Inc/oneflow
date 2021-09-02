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

#include "type_traits.h"

namespace oneflow {

namespace private_details {

template<template<typename> class Predicator>
struct StaticReduce {
  template<typename... Args>
  struct All;
  template<typename Void>
  struct All<Void> {
    static_assert(std::is_same<Void, void>::value, "");
    static constexpr bool value = true;
  };
  template<typename Void, typename T, typename... Args>
  struct All<Void, T, Args...> {
    static constexpr bool value = Predicator<T>::value && All<Void, Args...>::value;
  };

  template<typename... Args>
  struct Any;
  template<typename Void>
  struct Any<Void> {
    static_assert(std::is_same<Void, void>::value, "");
    static constexpr bool value = false;
  };
  template<typename Void, typename T, typename... Args>
  struct Any<Void, T, Args...> {
    static constexpr bool value = Predicator<T>::value || Any<Void, Args...>::value;
  };
};

}  // namespace private_details

template<template<typename> class Predicator, typename... Args>
struct StaticAll {
  static constexpr bool value =
      private_details::StaticReduce<Predicator>::template All<void, Args...>::value;
};

template<template<typename> class Predicator, typename... Args>
struct StaticAny {
  static constexpr bool value =
      private_details::StaticReduce<Predicator>::template Any<void, Args...>::value;
};

template<typename T>
struct IsOutArg {
  static constexpr bool value =
      (std::is_reference<T>::value
       && !std::is_const<typename std::remove_reference<T>::type>::value)
      || (std::is_pointer<T>::value
          && !std::is_const<typename std::remove_pointer<T>::type>::value);
};

template<typename T>
struct IsDecayedScalarType {
  static constexpr bool value = IsScalarType<typename std::decay<T>::type>::value;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STATIC_CHECK_H_
