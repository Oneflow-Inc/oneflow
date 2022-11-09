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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_BFLOAT16_H_
#define ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_BFLOAT16_H_

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#if CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

#define DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR(op)                \
  __device__ __forceinline__ __nv_bfloat16 operator op(const __nv_bfloat16& lh,   \
                                                       const __nv_bfloat16& rh) { \
    return __float2bfloat16(__bfloat162float(lh) op __bfloat162float(rh));        \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR(+)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR(-)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR(*)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR(/)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_OPERATOR

#define DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_FUNC(func)                       \
  __device__ __forceinline__ __nv_bfloat16 __h##func(const __nv_bfloat16 a,            \
                                                     const __nv_bfloat16 b) {          \
    return __float2bfloat16(__f##func##_rn(__bfloat162float(a), __bfloat162float(b))); \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_FUNC(add)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_FUNC(div)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_FUNC(mul)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_BINARY_FUNC(sub)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC

#define DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC(func)         \
  __device__ __forceinline__ __nv_bfloat162 __h##func##2(const __nv_bfloat162 a,   \
                                                         const __nv_bfloat162 b) { \
    __nv_bfloat162 ret;                                                            \
    ret.x = __h##func(a.x, b.x);                                                   \
    ret.y = __h##func(a.y, b.y);                                                   \
    return ret;                                                                    \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC(add)
DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC(div)
DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC(mul)
DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC(sub)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_BFLOAT162_ARITHMETIC_BINARY_FUNC

#define DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR(op)             \
  __device__ __forceinline__ __nv_bfloat16& operator op(__nv_bfloat16& lh,         \
                                                        const __nv_bfloat16& rh) { \
    float lhv = __bfloat162float(lh);                                              \
    lhv op __bfloat162float(rh);                                                   \
    lh = __float2bfloat16(lhv);                                                    \
    return lh;                                                                     \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR(+=)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR(-=)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR(*=)
DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR(/=)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_ARITHMETIC_ASSIGNMENT_OPERATOR

__device__ __forceinline__ __nv_bfloat16& operator++(__nv_bfloat16& h) {
  h = __float2bfloat16(__bfloat162float(h) + 1);
  return h;
}

__device__ __forceinline__ __nv_bfloat16& operator--(__nv_bfloat16& h) {
  h = __float2bfloat16(__bfloat162float(h) - 1);
  return h;
}

__device__ __forceinline__ __nv_bfloat16 operator++(__nv_bfloat16& h, int) {
  __nv_bfloat16 ret = h;
  h = __float2bfloat16(__bfloat162float(h) + 1);
  return ret;
}

__device__ __forceinline__ __nv_bfloat16 operator--(__nv_bfloat16& h, int) {
  __nv_bfloat16 ret = h;
  h = __float2bfloat16(__bfloat162float(h) - 1);
  return ret;
}

__device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16& h) { return h; }

__device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16& h) {
  return __float2bfloat16(-__bfloat162float(h));
}

__device__ __forceinline__ __nv_bfloat16 __hneg(const __nv_bfloat16 a) { return -a; }

#define DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(op)                                \
  __device__ __forceinline__ bool operator op(const __nv_bfloat16& lh, const __nv_bfloat16& rh) { \
    return __bfloat162float(lh) op __bfloat162float(rh);                                          \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(==)
DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(!=)
DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(>)
DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(<)
DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(>=)
DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR(<=)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_COMPARISON_BINARY_OPERATOR

__device__ __forceinline__ bool __heq(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a == b;
}
__device__ __forceinline__ bool __hge(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a >= b;
}
__device__ __forceinline__ bool __hgt(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a > b;
}
__device__ __forceinline__ bool __hle(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a <= b;
}
__device__ __forceinline__ bool __hlt(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a < b;
}
__device__ __forceinline__ bool __hne(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a != b;
}
__device__ __forceinline__ __nv_bfloat16 __hmax(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a > b ? a : b;
}
__device__ __forceinline__ __nv_bfloat16 __hmin(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return a > b ? a : b;
}

#define DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(func)                         \
  __device__ __forceinline__ __nv_bfloat16 h##func(const __nv_bfloat16 h) { \
    return __float2bfloat16(func##f(__bfloat162float(h)));                  \
  }

DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(cos)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(exp)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(exp10)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(exp2)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(log)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(log10)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(log2)

__device__ __forceinline__ __nv_bfloat16 hrcp(const __nv_bfloat16 h) {
  return __float2bfloat16(1.0f / __bfloat162float(h));
}

DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(rsqrt)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(sin)
DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC(sqrt)

#undef DEFINE_CUDA_PSEUDO_BFLOAT16_MATH_FUNC

#endif  // CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_BFLOAT16_H_
