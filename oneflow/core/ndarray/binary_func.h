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

#define ARITHMETIC_BINARY_FUNC_NAME_SEQ (Add)(Sub)(Mul)(Div)(Min)(Max)(FloorMod)(FMod)(Pow)
#define LOGICAL_BINARY_FUNC_NAME_SEQ (EQ)(NE)(GT)(GE)(LT)(LE)(AND)(OR)(XOR)

#define PREPEND_PREFIX_BINARY_FUNC(name) OF_PP_CAT(BinaryFunc, name)
#define ARITHMETIC_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#define LOGICAL_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, LOGICAL_BINARY_FUNC_NAME_SEQ)

#define REDUCE_BINARY_FUNC_NAME_SEQ (Sum)(Max)(Min)(Prod)(Any)(All)
#define ARITHMETIC_REDUCE_BINARY_FUNC_NAME_SEQ (Sum)(Max)(Min)(Prod)
#define LOGICAL_REDUCE_BINARY_FUNC_NAME_SEQ (Any)(All)
#define REDUCE_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ)
#define ARITHMETIC_REDUCE_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, ARITHMETIC_REDUCE_BINARY_FUNC_NAME_SEQ)
#define LOGICAL_REDUCE_BINARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, LOGICAL_REDUCE_BINARY_FUNC_NAME_SEQ)
#define NANSUM_REDUCE_BINARY_FUNC_SEQ OF_PP_SEQ_MAP(PREPEND_PREFIX_BINARY_FUNC, (NanSum))

#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)
template<template<typename> class BinaryFunc, typename T>
struct BinaryFuncTrait final {
  typedef typename std::remove_const<decltype(
      BinaryFunc<T>::Invoke(std::declval<const T>(), std::declval<const T>()))>::type return_type;
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
struct BinaryFuncNanSum final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) {
#if defined(__CUDACC__)
    if (isnan(x)) return isnan(y) ? T{0} : y;
    return isnan(y) ? x : x + y;
#else
    if (std::isnan(x)) return std::isnan(y) ? T{0} : y;
    return std::isnan(y) ? x : x + y;
#endif
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncNanSum);

template<typename T>
struct BinaryFuncAdd final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x + y; }
};
template<typename T>
struct BinaryFuncSum final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return BinaryFuncAdd<T>::Invoke(x, y); }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncAdd);

template<typename T>
struct BinaryFuncSub final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x - y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncSub);

template<typename T>
struct BinaryFuncMul final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x * y; }
};
template<>
struct BinaryFuncMul<bool> final {
  static OF_DEVICE_FUNC bool Invoke(const bool x, const bool y) { return x && y; }
};
template<typename T>
struct BinaryFuncProd final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return BinaryFuncMul<T>::Invoke(x, y); }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMul);

template<typename T>
struct BinaryFuncDiv final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x / y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncDiv);

template<typename T>
struct BinaryFuncFloorMod final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) {
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

template<>
struct BinaryFuncFloorMod<uint8_t> final {
  static OF_DEVICE_FUNC uint8_t Invoke(const uint8_t x, const uint8_t y) {
#if defined(__CUDACC__)
    uint8_t trunc_mod = x % y;
    return trunc_mod;
#else
    uint8_t trunc_mod = x % y;
    return trunc_mod;
#endif
  }
};

template<typename T>
struct BinaryFuncFMod final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) {
#if defined(__CUDACC__)
    T trunc_mod = x % y;
    return trunc_mod;
#else
    T trunc_mod = x % y;
    return trunc_mod;
#endif
  }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncFMod);

template<typename T>
struct BinaryFuncPow final {
  static OF_DEVICE_FUNC const T Invoke(const T x, const T y) {
#if defined(__CUDACC__)
    return powf(x, y);
#else
    return std::pow(x, y);
#endif
  }
};

template<>
struct BinaryFuncPow<bool> final {
  static OF_DEVICE_FUNC bool Invoke(const bool x, const bool y) {
#if defined(__CUDACC__)
    return static_cast<bool>(powf(static_cast<float>(x), static_cast<float>(y)));
#else
    return static_cast<bool>(std::pow(static_cast<float>(x), static_cast<float>(y)));
#endif
  }
};

SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncPow);

template<>
struct BinaryFuncPow<float16> final {
  static inline const float16 Invoke(const float16 x, const float16 y) {
    return static_cast<float16>(std::pow(static_cast<float>(x), static_cast<float>(y)));
  }
};

#if defined(__CUDACC__)
template<>
struct BinaryFuncPow<double> final {
  static OF_DEVICE_FUNC double Invoke(const double x, const double y) { return pow(x, y); }
};

template<>
struct BinaryFuncPow<float> final {
  static __device__ __forceinline__ float Invoke(const float x, const float y) {
    return powf(x, y);
  }
};

template<>
struct BinaryFuncPow<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __float2half(powf(__half2float(x), __half2float(y)));
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#endif  // defined(__CUDACC__)

template<typename T>
struct BinaryFuncFloorDiv final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) {
#if defined(__CUDACC__)
    return floor(fdividef(x, y));
#else
    return std::floor(x / y);
#endif
  }
};

template<typename T>
struct BinaryFuncMax final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x > y ? x : y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMax);

template<typename T>
struct BinaryFuncMin final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x < y ? x : y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncMin);

template<typename T>
struct BinaryFuncEQ final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x == y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncEQ);

template<typename T>
struct BinaryFuncNE final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x != y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncNE);

template<typename T>
struct BinaryFuncGT final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x > y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncGT);

template<typename T>
struct BinaryFuncGE final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x >= y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncGE);

template<typename T>
struct BinaryFuncLT final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x < y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncLT);

template<typename T>
struct BinaryFuncLE final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x <= y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncLE);

template<typename T>
struct BinaryFuncIEN final {
  // placeholder, no definition required, the type is only used to generate Op
};

template<typename T>
struct BinaryFuncINN final {
  // placeholder, no definition required, the type is only used to generate Op
};

template<typename T>
struct BinaryFuncAND final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x && y; }
};
template<typename T>
struct BinaryFuncAll final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return BinaryFuncAND<T>::Invoke(x, y); }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncAND);

template<typename T>
struct BinaryFuncOR final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return x || y; }
};
template<typename T>
struct BinaryFuncAny final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return BinaryFuncOR<T>::Invoke(x, y); }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncOR);

template<typename T>
struct BinaryFuncXOR final {
  static OF_DEVICE_FUNC bool Invoke(const T x, const T y) { return (!x) != (!y); }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncXOR);

template<typename T>
struct BinaryFuncBitwiseAnd final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x & y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncBitwiseAnd);

template<typename T>
struct BinaryFuncBitwiseOr final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x | y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncBitwiseOr);

template<typename T>
struct BinaryFuncBitwiseXor final {
  static OF_DEVICE_FUNC T Invoke(const T x, const T y) { return x ^ y; }
};
SPECIALIZE_CONST_TYPE_BINARY_FUNC(BinaryFuncBitwiseXor);

#if defined(__CUDACC__)
template<>
struct BinaryFuncAdd<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) { return __hadd(x, y); }
};

template<>
struct BinaryFuncNanSum<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
    if (isnan(__half2float(x))) return isnan(__half2float(y)) ? half(0.0) : y;
    return isnan(__half2float(y)) ? __hadd(x, y) : x;
  }
};

template<>
struct BinaryFuncSub<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hsub(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMul<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncDiv<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hdiv(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMax<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hgt(x, y) ? x : y;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};

template<>
struct BinaryFuncMin<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
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
  static __device__ __forceinline__ float Invoke(const float x, const float y) {
    const float trunc_mod = fmodf(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<double> final {
  static __device__ __forceinline__ double Invoke(const double x, const double y) {
    const double trunc_mod = fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
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
  static inline float Invoke(const float x, const float y) {
    const float trunc_mod = std::fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<double> final {
  static inline double Invoke(const double x, const double y) {
    const double trunc_mod = std::fmod(x, y);
    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? trunc_mod + y : trunc_mod;
  }
};

template<>
struct BinaryFuncFloorMod<float16> final {
  static inline float16 Invoke(const float16 x, const float16 y) {
    const float trunc_mod = std::fmod(static_cast<float>(x), static_cast<float>(y));
    return (trunc_mod != float(0)) && ((y < float(0)) != (trunc_mod < float(0)))
               ? static_cast<float16>(trunc_mod + y)
               : static_cast<float16>(trunc_mod);
  }
};

#endif  // defined(__CUDACC__)

#if defined(__CUDACC__)

template<>
struct BinaryFuncFMod<float> final {
  static __device__ __forceinline__ float Invoke(const float x, const float y) {
    const float trunc_mod = fmodf(x, y);
    return trunc_mod;
  }
};

template<>
struct BinaryFuncFMod<double> final {
  static __device__ __forceinline__ double Invoke(const double x, const double y) {
    const double trunc_mod = fmod(x, y);
    return trunc_mod;
  }
};

template<>
struct BinaryFuncFMod<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    const half trunc_mod = __float2half(fmodf(__half2float(x), __half2float(y)));
    return trunc_mod;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#else
template<>
struct BinaryFuncFMod<float> final {
  static inline float Invoke(const float x, const float y) {
    const float trunc_mod = std::fmod(x, y);
    return trunc_mod;
  }
};

template<>
struct BinaryFuncFMod<double> final {
  static inline double Invoke(const double x, const double y) {
    const double trunc_mod = std::fmod(x, y);
    return trunc_mod;
  }
};

template<>
struct BinaryFuncFMod<float16> final {
  static inline float16 Invoke(const float16 x, const float16 y) {
    const float trunc_mod = std::fmod(static_cast<float>(x), static_cast<float>(y));
    return static_cast<float16>(trunc_mod);
  }
};

#endif  // defined(__CUDACC__)

#if defined(__CUDACC__)

template<>
struct BinaryFuncFloorDiv<uint8_t> final {
  static __device__ __forceinline__ uint8_t Invoke(uint8_t x, uint8_t y) { return x / y; }
};

template<>
struct BinaryFuncFloorDiv<int8_t> final {
  static __device__ __forceinline__ int8_t Invoke(int8_t x, int8_t y) { return x / y; }
};

template<>
struct BinaryFuncFloorDiv<int32_t> final {
  static __device__ __forceinline__ int32_t Invoke(int32_t x, int32_t y) { return x / y; }
};

template<>
struct BinaryFuncFloorDiv<int64_t> final {
  static __device__ __forceinline__ int64_t Invoke(int64_t x, int64_t y) { return x / y; }
};

template<>
struct BinaryFuncFloorDiv<half> final {
  static __device__ __forceinline__ half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __float2half(floor(fdividef(__half2float(x), __half2float(y))));
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#else
template<>
struct BinaryFuncFloorDiv<float16> final {
  static inline float16 Invoke(float16 x, float16 y) {
    return static_cast<float16>(std::floor(static_cast<float>(x) / static_cast<float>(y)));
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
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncNanSum, GetZeroVal);
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
