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
// reference: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/NumericLimits.cuh
#pragma once
#include <limits.h>
#include <math.h>
#include <float.h>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"

// numeric_limits.cuh is a holder for numeric limits definitions of commonly used
// types. This header is very specific to ROCm HIP and may be removed in the future.

// The lower_bound and upper_bound constants are same as lowest and max for
// integral types, but are -inf and +inf for floating point types. They are
// useful in implementing min, max, etc.

namespace oneflow {
namespace detail {

#if defined(__CUDACC__)
#define OF_NUMERICS_FUNC static inline __host__ __device__
#else
#define OF_NUMERICS_FUNC static inline
#endif

template<typename T>
struct numeric_limits {};

// WARNING: the following oneflow::numeric_limits definitions are there only to support
//          HIP compilation for the moment. Use std::numeric_limits if you are not
//          compiling for ROCm.
//          from @colesbury: "The functions on numeric_limits aren't marked with
//          __device__ which is why they don't work with ROCm. CUDA allows them
//          because they're constexpr."

namespace {
// ROCm doesn't like INFINITY too.
constexpr double inf = INFINITY;
}  // namespace

template<>
struct numeric_limits<bool> {
  OF_NUMERICS_FUNC bool lowest() { return false; }
  OF_NUMERICS_FUNC bool max() { return true; }
  OF_NUMERICS_FUNC bool lower_bound() { return false; }
  OF_NUMERICS_FUNC bool upper_bound() { return true; }
};

template<>
struct numeric_limits<uint8_t> {
  OF_NUMERICS_FUNC uint8_t lowest() { return 0; }
  OF_NUMERICS_FUNC uint8_t max() { return UINT8_MAX; }
  OF_NUMERICS_FUNC uint8_t lower_bound() { return 0; }
  OF_NUMERICS_FUNC uint8_t upper_bound() { return UINT8_MAX; }
};

template<>
struct numeric_limits<int8_t> {
  OF_NUMERICS_FUNC int8_t lowest() { return INT8_MIN; }
  OF_NUMERICS_FUNC int8_t max() { return INT8_MAX; }
  OF_NUMERICS_FUNC int8_t lower_bound() { return INT8_MIN; }
  OF_NUMERICS_FUNC int8_t upper_bound() { return INT8_MAX; }
};

template<>
struct numeric_limits<int16_t> {
  OF_NUMERICS_FUNC int16_t lowest() { return INT16_MIN; }
  OF_NUMERICS_FUNC int16_t max() { return INT16_MAX; }
  OF_NUMERICS_FUNC int16_t lower_bound() { return INT16_MIN; }
  OF_NUMERICS_FUNC int16_t upper_bound() { return INT16_MAX; }
};

template<>
struct numeric_limits<int32_t> {
  OF_NUMERICS_FUNC int32_t lowest() { return INT32_MIN; }
  OF_NUMERICS_FUNC int32_t max() { return INT32_MAX; }
  OF_NUMERICS_FUNC int32_t lower_bound() { return INT32_MIN; }
  OF_NUMERICS_FUNC int32_t upper_bound() { return INT32_MAX; }
};

template<>
struct numeric_limits<int64_t> {
#ifdef _MSC_VER
  OF_NUMERICS_FUNC int64_t lowest() { return _I64_MIN; }
  OF_NUMERICS_FUNC int64_t max() { return _I64_MAX; }
  OF_NUMERICS_FUNC int64_t lower_bound() { return _I64_MIN; }
  OF_NUMERICS_FUNC int64_t upper_bound() { return _I64_MAX; }
#else
  OF_NUMERICS_FUNC int64_t lowest() { return INT64_MIN; }
  OF_NUMERICS_FUNC int64_t max() { return INT64_MAX; }
  OF_NUMERICS_FUNC int64_t lower_bound() { return INT64_MIN; }
  OF_NUMERICS_FUNC int64_t upper_bound() { return INT64_MAX; }
#endif
};

template<>
struct numeric_limits<float> {
  OF_NUMERICS_FUNC float lowest() { return -FLT_MAX; }
  OF_NUMERICS_FUNC float max() { return FLT_MAX; }
  OF_NUMERICS_FUNC float lower_bound() { return -static_cast<float>(inf); }
  OF_NUMERICS_FUNC float upper_bound() { return static_cast<float>(inf); }
};

#if defined(__CUDACC__)
static __device__ unsigned short int HALF_LOWEST = 0xfbff;
static __device__ unsigned short int HALF_MAX = 0x7bff;
static __device__ unsigned short int HALF_LOWER_BOUND = 0xfc00;
static __device__ unsigned short int HALF_UPPER_BOUND = 0x7c00;
template<>
struct numeric_limits<half> {
  static inline __device__ half lowest() { return *reinterpret_cast<const __half*>(&HALF_LOWEST); }
  static inline __device__ half max() { return *reinterpret_cast<const __half*>(&HALF_MAX); }
  static inline __device__ half lower_bound() {
    return *reinterpret_cast<const __half*>(&HALF_LOWER_BOUND);
  }
  static inline __device__ half upper_bound() {
    return *reinterpret_cast<const __half*>(&HALF_UPPER_BOUND);
  }
};

#if CUDA_VERSION >= 11000

static __device__ unsigned short int NV_BFLOAT16_LOWEST = 0xff7f;
static __device__ unsigned short int NV_BFLOAT16_MAX = 0x7f7f;
static __device__ unsigned short int NV_BFLOAT16_LOWER_BOUND = 0xff80;
static __device__ unsigned short int NV_BFLOAT16_UPPER_BOUND = 0x7f80;
template<>
struct numeric_limits<nv_bfloat16> {
  static inline __device__ nv_bfloat16 lowest() {
    return *reinterpret_cast<const __nv_bfloat16*>(&NV_BFLOAT16_LOWEST);
  }
  static inline __device__ nv_bfloat16 max() {
    return *reinterpret_cast<const __nv_bfloat16*>(&NV_BFLOAT16_MAX);
  }
  static inline __device__ nv_bfloat16 lower_bound() {
    return *reinterpret_cast<const __nv_bfloat16*>(&NV_BFLOAT16_LOWER_BOUND);
  }
  static inline __device__ nv_bfloat16 upper_bound() {
    return *reinterpret_cast<const __nv_bfloat16*>(&NV_BFLOAT16_UPPER_BOUND);
  }
};

#endif  // CUDA_VERSION >= 11000

#endif  // defined(__CUDACC__)

template<>
struct numeric_limits<double> {
  OF_NUMERICS_FUNC double lowest() { return -DBL_MAX; }
  OF_NUMERICS_FUNC double max() { return DBL_MAX; }
  OF_NUMERICS_FUNC double lower_bound() { return -inf; }
  OF_NUMERICS_FUNC double upper_bound() { return inf; }
};

}  // namespace detail
}  // namespace oneflow
