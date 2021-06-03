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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_HALF_H_
#define ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_HALF_H_

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530

#define DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR(op)                        \
  __device__ __forceinline__ __half operator op(const __half& lh, const __half& rh) { \
    return __float2half(__half2float(lh) op __half2float(rh));                        \
  }

DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR(+)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR(-)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR(*)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR(/)

#undef DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_OPERATOR

#define DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_FUNC(func)                    \
  __device__ __forceinline__ __half __h##func(const __half a, const __half b) { \
    return __float2half(__f##func##_rn(__half2float(a), __half2float(b)));      \
  }

DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_FUNC(add)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_FUNC(div)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_FUNC(mul)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_BINARY_FUNC(sub)

#undef DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC

#define DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC(func)                    \
  __device__ __forceinline__ __half2 __h##func##2(const __half2 a, const __half2 b) { \
    __half2 ret;                                                                      \
    ret.x = __h##func(a.x, b.x);                                                      \
    ret.y = __h##func(a.y, b.y);                                                      \
    return ret;                                                                       \
  }

DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC(add)
DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC(div)
DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC(mul)
DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC(sub)

#undef DEFINE_CUDA_PSEUDO_HALF_HALF2_ARITHMETIC_BINARY_FUNC

#define DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR(op)               \
  __device__ __forceinline__ __half& operator op(__half& lh, const __half& rh) { \
    float lhv = __half2float(lh);                                                \
    lhv op __half2float(rh);                                                     \
    lh = __float2half(lhv);                                                      \
    return lh;                                                                   \
  }

DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR(+=)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR(-=)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR(*=)
DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR(/=)

#undef DEFINE_CUDA_PSEUDO_HALF_ARITHMETIC_ASSIGNMENT_OPERATOR

__device__ __forceinline__ __half& operator++(__half& h) {
  h = __float2half(__half2float(h) + 1);
  return h;
}

__device__ __forceinline__ __half& operator--(__half& h) {
  h = __float2half(__half2float(h) - 1);
  return h;
}

__device__ __forceinline__ __half operator++(__half& h, int) {
  __half ret = h;
  h = __float2half(__half2float(h) + 1);
  return ret;
}

__device__ __forceinline__ __half operator--(__half& h, int) {
  __half ret = h;
  h = __float2half(__half2float(h) - 1);
  return ret;
}

__device__ __forceinline__ __half operator+(const __half& h) { return h; }

__device__ __forceinline__ __half operator-(const __half& h) {
  return __float2half(-__half2float(h));
}

__device__ __forceinline__ __half __hneg(const __half a) { return -a; }

#define DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(op)                      \
  __device__ __forceinline__ bool operator op(const __half& lh, const __half& rh) { \
    return __half2float(lh) op __half2float(rh);                                    \
  }

DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(==)
DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(!=)
DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(>)
DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(<)
DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(>=)
DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR(<=)

#undef DEFINE_CUDA_PSEUDO_HALF_COMPARISON_BINARY_OPERATOR

__device__ __forceinline__ bool __heq(const __half a, const __half b) { return a == b; }
__device__ __forceinline__ bool __hge(const __half a, const __half b) { return a >= b; }
__device__ __forceinline__ bool __hgt(const __half a, const __half b) { return a > b; }
__device__ __forceinline__ bool __hle(const __half a, const __half b) { return a <= b; }
__device__ __forceinline__ bool __hlt(const __half a, const __half b) { return a < b; }
__device__ __forceinline__ bool __hne(const __half a, const __half b) { return a != b; }
__device__ __forceinline__ __half __hmax(const __half a, const __half b) { return a > b ? a : b; }
__device__ __forceinline__ __half __hmin(const __half a, const __half b) { return a > b ? a : b; }

#define DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(func)               \
  __device__ __forceinline__ __half h##func(const __half h) { \
    return __float2half(func##f(__half2float(h)));            \
  }

DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(cos)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(exp)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(exp10)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(exp2)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(log)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(log10)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(log2)

__device__ __forceinline__ __half hrcp(const __half h) {
  return __float2half(1.0f / __half2float(h));
}

DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(rsqrt)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(sin)
DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC(sqrt)

#undef DEFINE_CUDA_PSEUDO_HALF_MATH_FUNC

#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_PSEUDO_HALF_H_
