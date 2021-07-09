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
#pragma once
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <half.hpp>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"

// numeric_limits.cuh is a holder for numeric limits definitions of commonly used
// types. This header is very specific to ROCm HIP and may be removed in the future.

// The lower_bound and upper_bound constants are same as lowest and max for
// integral types, but are -inf and +inf for floating point types. They are
// useful in implementing min, max, etc.

namespace oneflow {

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
  static inline __host__ __device__ bool lowest() { return false; }
  static inline __host__ __device__ bool max() { return true; }
  static inline __host__ __device__ bool lower_bound() { return false; }
  static inline __host__ __device__ bool upper_bound() { return true; }
};

template<>
struct numeric_limits<uint8_t> {
  static inline __host__ __device__ uint8_t lowest() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UINT8_MAX; }
  static inline __host__ __device__ uint8_t lower_bound() { return 0; }
  static inline __host__ __device__ uint8_t upper_bound() { return UINT8_MAX; }
};

template<>
struct numeric_limits<int8_t> {
  static inline __host__ __device__ int8_t lowest() { return INT8_MIN; }
  static inline __host__ __device__ int8_t max() { return INT8_MAX; }
  static inline __host__ __device__ int8_t lower_bound() { return INT8_MIN; }
  static inline __host__ __device__ int8_t upper_bound() { return INT8_MAX; }
};

template<>
struct numeric_limits<int16_t> {
  static inline __host__ __device__ int16_t lowest() { return INT16_MIN; }
  static inline __host__ __device__ int16_t max() { return INT16_MAX; }
  static inline __host__ __device__ int16_t lower_bound() { return INT16_MIN; }
  static inline __host__ __device__ int16_t upper_bound() { return INT16_MAX; }
};

template<>
struct numeric_limits<int32_t> {
  static inline __host__ __device__ int32_t lowest() { return INT32_MIN; }
  static inline __host__ __device__ int32_t max() { return INT32_MAX; }
  static inline __host__ __device__ int32_t lower_bound() { return INT32_MIN; }
  static inline __host__ __device__ int32_t upper_bound() { return INT32_MAX; }
};

template<>
struct numeric_limits<int64_t> {
#ifdef _MSC_VER
  static inline __host__ __device__ int64_t lowest() { return _I64_MIN; }
  static inline __host__ __device__ int64_t max() { return _I64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return _I64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return _I64_MAX; }
#else
  static inline __host__ __device__ int64_t lowest() { return INT64_MIN; }
  static inline __host__ __device__ int64_t max() { return INT64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return INT64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return INT64_MAX; }
#endif
};

template<>
struct numeric_limits<float> {
  static inline __host__ __device__ float lowest() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }
  static inline __host__ __device__ float lower_bound() { return -static_cast<float>(inf); }
  static inline __host__ __device__ float upper_bound() { return static_cast<float>(inf); }
};

template<>
struct numeric_limits<double> {
  static inline __host__ __device__ double lowest() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }
  static inline __host__ __device__ double lower_bound() { return -inf; }
  static inline __host__ __device__ double upper_bound() { return inf; }
};

}  // namespace oneflow
