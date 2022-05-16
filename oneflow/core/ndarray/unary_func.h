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
#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#if defined(__CUDACC__)
#include <cuda_fp16.hpp>
#endif
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_NAME_SEQ (Identity)(Negative)(Exp)

#define PREPEND_PREFIX_UNARY_FUNC(name) OF_PP_CAT(UnaryFunc, name)
#define ARITHMETIC_UNARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)

template<template<typename> class UnaryFunc, typename T>
struct UnaryFuncTrait final {
  typedef typename std::remove_const<decltype(UnaryFunc<T>::Invoke(*(const T*)nullptr))>::type
      return_type;
};

#define SPECIALIZE_CONST_TYPE_UNARY_FUNC(func_struct)                                     \
  template<typename T>                                                                    \
  struct func_struct<const T> final {                                                     \
    static OF_DEVICE_FUNC const T Invoke(const T x) { return func_struct<T>::Invoke(x); } \
  }

template<typename T>
struct UnaryFuncIdentity final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return x; }
};

template<typename T>
struct UnaryFuncNegative final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return -x; }
};
SPECIALIZE_CONST_TYPE_UNARY_FUNC(UnaryFuncNegative);

template<typename T>
struct UnaryFuncExp final {
  static OF_DEVICE_FUNC const T Invoke(const T x) {
#if defined(__CUDA_ARCH__)
    if (std::is_same<T, double>::value) {
      return static_cast<T>(exp(static_cast<double>(x)));
    } else {
      return static_cast<T>(exp(static_cast<float>(x)));
    }
#else
    return std::exp(x);
#endif  // defined(__CUDA_ARCH__)
  }
};

template<>
struct UnaryFuncExp<bool> final {
  static OF_DEVICE_FUNC bool Invoke(const bool x) {
#if defined(__CUDA_ARCH__)
    return static_cast<bool>(exp(static_cast<float>(x)));
#else
    return static_cast<bool>(std::exp(static_cast<float>(x)));
#endif  // defined(__CUDA_ARCH__)
  }
};
SPECIALIZE_CONST_TYPE_UNARY_FUNC(UnaryFuncExp);

template<>
struct UnaryFuncExp<float16> final {
  static OF_DEVICE_FUNC const float16 Invoke(const float16 x) {
#if defined(__CUDA_ARCH__)
    half res = static_cast<half>(exp(static_cast<float>(*reinterpret_cast<const half*>(&x))));
    return *reinterpret_cast<float16*>(&res);
#else
    return float16(std::exp(static_cast<float>(x)));
#endif  // defined(__CUDA_ARCH__)
  }
};
#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)

#if defined(__CUDACC__)
template<>
struct UnaryFuncNegative<half> final {
  static __device__ __forceinline__ const half Invoke(const half x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hneg(x);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct UnaryFuncExp<half> final {
  static __device__ __forceinline__ const half Invoke(const half x) {
    return __float2half(std::exp(__half2float(x)));
  }
};
#endif

template<typename T>
struct UnaryFuncLogicalNot final {
  static OF_DEVICE_FUNC bool Invoke(const T x) { return !x; }
};
SPECIALIZE_CONST_TYPE_UNARY_FUNC(UnaryFuncLogicalNot);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
