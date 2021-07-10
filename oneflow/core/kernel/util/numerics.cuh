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
// reference: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCNumerics.cuh
#ifndef ONEFLOW_CORE_KERNEL_UTIL_NUMERICS_H
#define ONEFLOW_CORE_KERNEL_UTIL_NUMERICS_H
#pragma once

#include <limits.h>
#include <math.h>
#include <float.h>
#include <cstdlib>
#include <assert.h>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"

namespace oneflow {

template<typename T>
struct numerics {};

template<typename T>
static inline __host__ __device__ T powi(T a, T b) {
  assert(numerics<T>::ge(b, 0));
  T result = 1;
  while (b) {
    if (b & 1) { result *= a; }
    b /= 2;
    a *= a;
  }
  return result;
}

template<>
struct numerics<uint8_t> {
  static inline __host__ __device__ uint8_t min() {
    return oneflow::numeric_limits<uint8_t>::lowest();
  }
  static inline __host__ __device__ uint8_t max() {
    return oneflow::numeric_limits<uint8_t>::max();
  }
  static inline __host__ __device__ uint8_t lower_bound() {
    return oneflow::numeric_limits<uint8_t>::lower_bound();
  }
  static inline __host__ __device__ uint8_t upper_bound() {
    return oneflow::numeric_limits<uint8_t>::upper_bound();
  }

  static inline __host__ __device__ bool lt(uint8_t a, uint8_t b) { return a < b; }
  static inline __host__ __device__ bool le(uint8_t a, uint8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(uint8_t a, uint8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(uint8_t a, uint8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(uint8_t a, uint8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(uint8_t a, uint8_t b) { return a != b; }

  static inline __host__ __device__ uint8_t add(uint8_t a, uint8_t b) { return a + b; }
  static inline __host__ __device__ uint8_t mul(uint8_t a, uint8_t b) { return a * b; }
  static inline __host__ __device__ uint8_t sub(uint8_t a, uint8_t b) { return a - b; }
  static inline __host__ __device__ uint8_t div(uint8_t a, uint8_t b) { return a / b; }
  static inline __host__ __device__ uint8_t pow(uint8_t a, uint8_t b) {
    return powi<uint8_t>(a, b);
  }
  static inline __host__ __device__ bool isnan(uint8_t a) { return false; }
  static inline __host__ __device__ bool isinf(uint8_t a) { return false; }
};

#ifdef _MSC_VER
// Suppress warning C4804: '/': unsafe use of type 'bool' in operation
#pragma warning(push)
#pragma warning(disable : 4804)
#endif

template<>
struct numerics<bool> {
  static inline __host__ __device__ bool min() { return oneflow::numeric_limits<bool>::lowest(); }
  static inline __host__ __device__ bool max() { return oneflow::numeric_limits<bool>::max(); }
  static inline __host__ __device__ bool lower_bound() {
    return oneflow::numeric_limits<bool>::lower_bound();
  }
  static inline __host__ __device__ bool upper_bound() {
    return oneflow::numeric_limits<bool>::upper_bound();
  }

  static inline __host__ __device__ bool lt(bool a, bool b) { return a < b; }
  static inline __host__ __device__ bool le(bool a, bool b) { return a <= b; }
  static inline __host__ __device__ bool gt(bool a, bool b) { return a > b; }
  static inline __host__ __device__ bool ge(bool a, bool b) { return a >= b; }
  static inline __host__ __device__ bool eq(bool a, bool b) { return a == b; }
  static inline __host__ __device__ bool ne(bool a, bool b) { return a != b; }
  static inline __host__ __device__ bool add(bool a, bool b) { return a + b; }
  static inline __host__ __device__ bool mul(bool a, bool b) { return a && b; }
  static inline __host__ __device__ bool sub(bool a, bool b) { return a - b; }
  static inline __host__ __device__ bool div(bool a, bool b) { return a / b; }
  static inline __host__ __device__ bool isnan(bool a) { return false; }
  static inline __host__ __device__ bool isinf(bool a) { return false; }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template<>
struct numerics<int8_t> {
  static inline __host__ __device__ int8_t min() {
    return oneflow::numeric_limits<int8_t>::lowest();
  }
  static inline __host__ __device__ int8_t max() { return oneflow::numeric_limits<int8_t>::max(); }
  static inline __host__ __device__ int8_t lower_bound() {
    return oneflow::numeric_limits<int8_t>::lower_bound();
  }
  static inline __host__ __device__ int8_t upper_bound() {
    return oneflow::numeric_limits<int8_t>::upper_bound();
  }

  static inline __host__ __device__ bool lt(int8_t a, int8_t b) { return a < b; }
  static inline __host__ __device__ bool le(int8_t a, int8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int8_t a, int8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int8_t a, int8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int8_t a, int8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int8_t a, int8_t b) { return a != b; }

  static inline __host__ __device__ int8_t add(int8_t a, int8_t b) { return a + b; }
  static inline __host__ __device__ int8_t mul(int8_t a, int8_t b) { return a * b; }
  static inline __host__ __device__ int8_t sub(int8_t a, int8_t b) { return a - b; }
  static inline __host__ __device__ int8_t div(int8_t a, int8_t b) { return a / b; }
  static inline __host__ __device__ int8_t pow(int8_t a, int8_t b) { return powi<int8_t>(a, b); }
  static inline __host__ __device__ bool isnan(int8_t a) { return false; }
  static inline __host__ __device__ bool isinf(int8_t a) { return false; }
};

template<>
struct numerics<int16_t> {
  static inline __host__ __device__ int16_t min() {
    return oneflow::numeric_limits<int16_t>::lowest();
  }
  static inline __host__ __device__ int16_t max() {
    return oneflow::numeric_limits<int16_t>::max();
  }
  static inline __host__ __device__ int16_t lower_bound() {
    return oneflow::numeric_limits<int16_t>::lower_bound();
  }
  static inline __host__ __device__ int16_t upper_bound() {
    return oneflow::numeric_limits<int16_t>::upper_bound();
  }

  static inline __host__ __device__ bool lt(int16_t a, int16_t b) { return a < b; }
  static inline __host__ __device__ bool le(int16_t a, int16_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int16_t a, int16_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int16_t a, int16_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int16_t a, int16_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int16_t a, int16_t b) { return a != b; }

  static inline __host__ __device__ int16_t add(int16_t a, int16_t b) { return a + b; }
  static inline __host__ __device__ int16_t mul(int16_t a, int16_t b) { return a * b; }
  static inline __host__ __device__ int16_t sub(int16_t a, int16_t b) { return a - b; }
  static inline __host__ __device__ int16_t div(int16_t a, int16_t b) { return a / b; }
  static inline __host__ __device__ int16_t pow(int16_t a, int16_t b) {
    return powi<int16_t>(a, b);
  }
  static inline __host__ __device__ bool isnan(int16_t a) { return false; }
  static inline __host__ __device__ bool isinf(int16_t a) { return false; }
};

template<>
struct numerics<int32_t> {
  static inline __host__ __device__ int32_t min() {
    return oneflow::numeric_limits<int32_t>::lowest();
  }
  static inline __host__ __device__ int32_t max() {
    return oneflow::numeric_limits<int32_t>::max();
  }
  static inline __host__ __device__ int32_t lower_bound() {
    return oneflow::numeric_limits<int32_t>::lower_bound();
  }
  static inline __host__ __device__ int32_t upper_bound() {
    return oneflow::numeric_limits<int32_t>::upper_bound();
  }

  static inline __host__ __device__ bool lt(int32_t a, int32_t b) { return a < b; }
  static inline __host__ __device__ bool le(int32_t a, int32_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int32_t a, int32_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int32_t a, int32_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int32_t a, int32_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int32_t a, int32_t b) { return a != b; }

  static inline __host__ __device__ int32_t add(int32_t a, int32_t b) { return a + b; }
  static inline __host__ __device__ int32_t mul(int32_t a, int32_t b) { return a * b; }
  static inline __host__ __device__ int32_t sub(int32_t a, int32_t b) { return a - b; }
  static inline __host__ __device__ int32_t div(int32_t a, int32_t b) { return a / b; }
  static inline __host__ __device__ int32_t pow(int32_t a, int32_t b) {
    return powi<int32_t>(a, b);
  }
  static inline __host__ __device__ bool isnan(int32_t a) { return false; }
  static inline __host__ __device__ bool isinf(int32_t a) { return false; }
};

template<>
struct numerics<int64_t> {
  static inline __host__ __device__ int64_t min() {
    return oneflow::numeric_limits<int64_t>::lowest();
  }
  static inline __host__ __device__ int64_t max() {
    return oneflow::numeric_limits<int64_t>::max();
  }
  static inline __host__ __device__ int64_t lower_bound() {
    return oneflow::numeric_limits<int64_t>::lower_bound();
  }
  static inline __host__ __device__ int64_t upper_bound() {
    return oneflow::numeric_limits<int64_t>::upper_bound();
  }

  static inline __host__ __device__ bool lt(int64_t a, int64_t b) { return a < b; }
  static inline __host__ __device__ bool le(int64_t a, int64_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int64_t a, int64_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int64_t a, int64_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int64_t a, int64_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int64_t a, int64_t b) { return a != b; }

  static inline __host__ __device__ int64_t add(int64_t a, int64_t b) { return a + b; }
  static inline __host__ __device__ int64_t mul(int64_t a, int64_t b) { return a * b; }
  static inline __host__ __device__ int64_t sub(int64_t a, int64_t b) { return a - b; }
  static inline __host__ __device__ int64_t div(int64_t a, int64_t b) { return a / b; };
  static inline __host__ __device__ int64_t pow(int64_t a, int64_t b) {
    return powi<int64_t>(a, b);
  }
  static inline __host__ __device__ bool isnan(int64_t a) { return false; }
  static inline __host__ __device__ bool isinf(int64_t a) { return false; }
};

// DEPRECATED: use math functions from std and cuda math API (if needed)
template<>
struct numerics<float> {
  static inline __host__ __device__ float min() { return oneflow::numeric_limits<float>::lowest(); }
  static inline __host__ __device__ float max() { return oneflow::numeric_limits<float>::max(); }
  static inline __host__ __device__ float lower_bound() {
    return oneflow::numeric_limits<float>::lower_bound();
  }
  static inline __host__ __device__ float upper_bound() {
    return oneflow::numeric_limits<float>::upper_bound();
  }

  static inline __host__ __device__ bool lt(float a, float b) { return a < b; }
  static inline __host__ __device__ bool le(float a, float b) { return a <= b; }
  static inline __host__ __device__ bool gt(float a, float b) { return a > b; }
  static inline __host__ __device__ bool ge(float a, float b) { return a >= b; }
  static inline __host__ __device__ bool eq(float a, float b) { return a == b; }
  static inline __host__ __device__ bool ne(float a, float b) { return a != b; }

  static inline __host__ __device__ float sqrt(float a) { return sqrtf(a); }
  static inline __host__ __device__ float atan(float a) { return atanf(a); }
  static inline __host__ __device__ float add(float a, float b) { return a + b; }
  static inline __host__ __device__ float div(float a, float b) { return a / b; }
  static inline __host__ __device__ float mul(float a, float b) { return a * b; }
  static inline __host__ __device__ float sub(float a, float b) { return a - b; }
  static inline __host__ __device__ float pow(float a, float b) { return powf(a, b); }
  static inline __host__ __device__ bool isnan(float a) { return ::isnan(a); }
  static inline __host__ __device__ bool isinf(float a) { return ::isinf(a); }
};

template<>
struct numerics<double> {
  static inline __host__ __device__ double min() {
    return oneflow::numeric_limits<double>::lowest();
  }
  static inline __host__ __device__ double max() { return oneflow::numeric_limits<double>::max(); }
  static inline __host__ __device__ double lower_bound() {
    return oneflow::numeric_limits<double>::lower_bound();
  }
  static inline __host__ __device__ double upper_bound() {
    return oneflow::numeric_limits<double>::upper_bound();
  }

  static inline __host__ __device__ bool lt(double a, double b) { return a < b; }
  static inline __host__ __device__ bool le(double a, double b) { return a <= b; }
  static inline __host__ __device__ bool gt(double a, double b) { return a > b; }
  static inline __host__ __device__ bool ge(double a, double b) { return a >= b; }
  static inline __host__ __device__ bool eq(double a, double b) { return a == b; }
  static inline __host__ __device__ bool ne(double a, double b) { return a != b; }

  static inline __host__ __device__ double sqrt(double a) { return ::sqrt(a); }
  static inline __host__ __device__ double atan(double a) { return ::atan(a); }
  static inline __host__ __device__ double add(double a, double b) { return a + b; }
  static inline __host__ __device__ double div(double a, double b) { return a / b; }
  static inline __host__ __device__ double mul(double a, double b) { return a * b; }
  static inline __host__ __device__ double sub(double a, double b) { return a - b; }
  static inline __host__ __device__ double pow(double a, double b) { return ::pow(a, b); }
  static inline __host__ __device__ bool isnan(double a) { return ::isnan(a); }
  static inline __host__ __device__ bool isinf(double a) { return ::isinf(a); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_NUMERICS_H
