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
#ifndef ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_

#include <cstdint>
#include <climits>
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"
namespace oneflow {

#define ARITHMETIC_BINARY_FUNC_NAME_SEQ (Add)(Sub)(Mul)(Div)(Min)(Max)(FloorMod)
#define LOGICAL_BINARY_FUNC_NAME_SEQ (EQ)(NE)(GT)(GE)(LT)(LE)(AND)

#define PREPEND_PREFIX_BINARY_FUNC(name) OF_PP_CAT(BinaryFunc, name)
#define ARITHMETIC_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#define LOGICAL_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, LOGICAL_BINARY_FUNC_NAME_SEQ)

#define BINARY_FUNC_NAME_SEQ ARITHMETIC_BINARY_FUNC_NAME_SEQ LOGICAL_BINARY_FUNC_NAME_SEQ
#define BINARY_FUNC_SEQ ARITHMETIC_BINARY_FUNC_SEQ LOGICAL_BINARY_FUNC_SEQ

#define REDUCE_BINARY_FUNC_NAME_SEQ (Sum)(Max)(Min)(Prod)(Any)(All)
#define REDUCE_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ)

template<template<typename> class BinaryFunc, typename T>
struct BinaryFuncTrait final {
  typedef typename std::remove_const<decltype(
      BinaryFunc<T>::Invoke(*(const T*)nullptr, *(const T*)nullptr))>::type return_type;
};

#define SPECIALIZE_CONST_TYPE_BINARY_FUNC(func_struct)                                        \
  template<typename T>                                                                        \
  struct func_struct<const T> final {                                                         \
    static OF_DEVICE_FUNC const typename BinaryFuncTrait<func_struct, T>::return_type Invoke( \
        const T x, const T y) {                                                               \
      return func_struct<T>::Invoke(x, y);                                                    \
    }                                                                                         \
  }

template<typename T>
struct BinaryFuncAdd final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x + y; }
};
template<typename T>
struct BinaryFuncSum final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) {
    return BinaryFuncAdd<T>::Invoke(x, y);
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncAdd);

template<typename T>
struct BinaryFuncSub final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x - y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncSub);

template<typename T>
struct BinaryFuncMul final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x * y; }
};
template<typename T>
struct BinaryFuncProd final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) {
    return BinaryFuncMul<T>::Invoke(x, y);
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMul);

template<typename T>
struct BinaryFuncDiv final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x / y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncDiv);

template<typename T>
struct BinaryFuncFloorMod final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) {
#if defined(__CUDACC__)
    T trunc_mod = x % y;
    return (trunc_mod != T(0)) && ((y < T(0)) != (trunc_mod < T(0))) ? trunc_mod + y : trunc_mod;
#else
    T trunc_mod = x % y;
    return (trunc_mod != T(0)) && ((y < T(0)) != (trunc_mod < T(0))) ? trunc_mod + y : trunc_mod;
#endif
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncFloorMod);

template<typename T>
struct BinaryFuncMax final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x > y ? x : y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMax);

template<typename T>
struct BinaryFuncMin final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x < y ? x : y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMin);

template<typename T>
struct BinaryFuncEQ final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x == y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncEQ);

template<typename T>
struct BinaryFuncNE final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x != y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncNE);

template<typename T>
struct BinaryFuncGT final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x > y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncGT);

template<typename T>
struct BinaryFuncGE final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x >= y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncGE);

template<typename T>
struct BinaryFuncLT final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x < y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncLT);

template<typename T>
struct BinaryFuncLE final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x <= y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncLE);

template<typename T>
struct BinaryFuncAND final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x && y; }
};
template<typename T>
struct BinaryFuncAll final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) {
    return BinaryFuncAND<T>::Invoke(x, y);
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncAND);

template<typename T>
struct BinaryFuncAny final {
  static OF_DEVICE_FUNC const int8_t Invoke(const T x, const T y) { return x || y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncAny);

#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)

#if defined(__CUDACC__)

template<>
struct BinaryFuncAdd<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
    return __hadd(x, y);
  }
};

template<>
struct BinaryFuncSub<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hsub(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMul<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncDiv<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hdiv(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMax<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hgt(x, y) ? x : y;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMin<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hlt(x, y) ? x : y;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

#endif  // defined(__CUDACC__)

#if defined(__CUDACC__)

template<>
struct BinaryFuncFloorMod<float> final {
  static __device__ __forceinline__ const float Invoke(const float x, const float y) {
    const float trunc_mod = fmodf(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<double> final {
  static __device__ __forceinline__ const double Invoke(const double x, const double y) {
    const double trunc_mod = fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<half> final {
  static __device__ __forceinline__ const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    const half trunc_mod = __float2half(fmodf(__half2float(x), __half2float(y)));
    return __hne(trunc_mod, GetZeroVal<half>())
                   && __hlt(y, GetZeroVal<half>()) != __hlt(trunc_mod, half(0))
               ? trunc_mod + y
               : trunc_mod;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

#else

template<>
struct BinaryFuncFloorMod<float> final {
  static inline const float Invoke(const float x, const float y) {
    const float trunc_mod = std::fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<double> final {
  static inline const double Invoke(const double x, const double y) {
    const double trunc_mod = std::fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<float16> final {
  static inline const float16 Invoke(const float16 x, const float16 y) {
    const float trunc_mod = std::fmod(static_cast<float>(x), static_cast<float>(y));
    return (trunc_mod != float(0)) && ((y < float(0)) != (trunc_mod < float(0)))
               ? static_cast<float16>(trunc_mod + y)
               : static_cast<float16>(trunc_mod);
  }
};

#endif  // defined(__CUDACC__)

template<typename T, template<typename> class binary_func>
struct UnitOfBinaryFunc;

#define SPECIALIZE_UNIT_OF_BINARY_FUNC(binary_func, get_val) \
  template<typename T>                                       \
  struct UnitOfBinaryFunc<T, binary_func> final {            \
    static OF_DEVICE_FUNC T Val() { return get_val<T>(); }   \
  };
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAdd, GetZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncSum, GetZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMul, GetOneVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncProd, GetOneVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMax, GetMinVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMin, GetMaxVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAny, GetZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAll, GetOneVal);
#undef SPECIALIZE_UNIT_OF_BINARY_FUNC

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
