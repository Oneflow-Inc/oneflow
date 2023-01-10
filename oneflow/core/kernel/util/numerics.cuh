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
namespace detail {

template<typename T>
struct numerics {};

template<typename T>
OF_NUMERICS_FUNC T powi(T a, T b) {
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
  OF_NUMERICS_FUNC uint8_t min() { return detail::numeric_limits<uint8_t>::lowest(); }
  OF_NUMERICS_FUNC uint8_t max() { return detail::numeric_limits<uint8_t>::max(); }
  OF_NUMERICS_FUNC uint8_t lower_bound() { return detail::numeric_limits<uint8_t>::lower_bound(); }
  OF_NUMERICS_FUNC uint8_t upper_bound() { return detail::numeric_limits<uint8_t>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(uint8_t a, uint8_t b) { return a < b; }
  OF_NUMERICS_FUNC bool le(uint8_t a, uint8_t b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(uint8_t a, uint8_t b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(uint8_t a, uint8_t b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(uint8_t a, uint8_t b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(uint8_t a, uint8_t b) { return a != b; }

  OF_NUMERICS_FUNC uint8_t add(uint8_t a, uint8_t b) { return a + b; }
  OF_NUMERICS_FUNC uint8_t mul(uint8_t a, uint8_t b) { return a * b; }
  OF_NUMERICS_FUNC uint8_t sub(uint8_t a, uint8_t b) { return a - b; }
  OF_NUMERICS_FUNC uint8_t div(uint8_t a, uint8_t b) { return a / b; }
  OF_NUMERICS_FUNC uint8_t pow(uint8_t a, uint8_t b) { return powi<uint8_t>(a, b); }
  OF_NUMERICS_FUNC bool isnan(uint8_t a) { return false; }
  OF_NUMERICS_FUNC bool isinf(uint8_t a) { return false; }
};

#ifdef _MSC_VER
// Suppress warning C4804: '/': unsafe use of type 'bool' in operation
#pragma warning(push)
#pragma warning(disable : 4804)
#endif

template<>
struct numerics<bool> {
  OF_NUMERICS_FUNC bool min() { return detail::numeric_limits<bool>::lowest(); }
  OF_NUMERICS_FUNC bool max() { return detail::numeric_limits<bool>::max(); }
  OF_NUMERICS_FUNC bool lower_bound() { return detail::numeric_limits<bool>::lower_bound(); }
  OF_NUMERICS_FUNC bool upper_bound() { return detail::numeric_limits<bool>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(bool a, bool b) { return a < b; }
  OF_NUMERICS_FUNC bool le(bool a, bool b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(bool a, bool b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(bool a, bool b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(bool a, bool b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(bool a, bool b) { return a != b; }
  OF_NUMERICS_FUNC bool add(bool a, bool b) { return a + b; }
  OF_NUMERICS_FUNC bool mul(bool a, bool b) { return a && b; }
  OF_NUMERICS_FUNC bool sub(bool a, bool b) { return a - b; }
  OF_NUMERICS_FUNC bool div(bool a, bool b) { return a / b; }
  OF_NUMERICS_FUNC bool isnan(bool a) { return false; }
  OF_NUMERICS_FUNC bool isinf(bool a) { return false; }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template<>
struct numerics<int8_t> {
  OF_NUMERICS_FUNC int8_t min() { return detail::numeric_limits<int8_t>::lowest(); }
  OF_NUMERICS_FUNC int8_t max() { return detail::numeric_limits<int8_t>::max(); }
  OF_NUMERICS_FUNC int8_t lower_bound() { return detail::numeric_limits<int8_t>::lower_bound(); }
  OF_NUMERICS_FUNC int8_t upper_bound() { return detail::numeric_limits<int8_t>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(int8_t a, int8_t b) { return a < b; }
  OF_NUMERICS_FUNC bool le(int8_t a, int8_t b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(int8_t a, int8_t b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(int8_t a, int8_t b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(int8_t a, int8_t b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(int8_t a, int8_t b) { return a != b; }

  OF_NUMERICS_FUNC int8_t add(int8_t a, int8_t b) { return a + b; }
  OF_NUMERICS_FUNC int8_t mul(int8_t a, int8_t b) { return a * b; }
  OF_NUMERICS_FUNC int8_t sub(int8_t a, int8_t b) { return a - b; }
  OF_NUMERICS_FUNC int8_t div(int8_t a, int8_t b) { return a / b; }
  OF_NUMERICS_FUNC int8_t pow(int8_t a, int8_t b) { return powi<int8_t>(a, b); }
  OF_NUMERICS_FUNC bool isnan(int8_t a) { return false; }
  OF_NUMERICS_FUNC bool isinf(int8_t a) { return false; }
};

template<>
struct numerics<int16_t> {
  OF_NUMERICS_FUNC int16_t min() { return detail::numeric_limits<int16_t>::lowest(); }
  OF_NUMERICS_FUNC int16_t max() { return detail::numeric_limits<int16_t>::max(); }
  OF_NUMERICS_FUNC int16_t lower_bound() { return detail::numeric_limits<int16_t>::lower_bound(); }
  OF_NUMERICS_FUNC int16_t upper_bound() { return detail::numeric_limits<int16_t>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(int16_t a, int16_t b) { return a < b; }
  OF_NUMERICS_FUNC bool le(int16_t a, int16_t b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(int16_t a, int16_t b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(int16_t a, int16_t b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(int16_t a, int16_t b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(int16_t a, int16_t b) { return a != b; }

  OF_NUMERICS_FUNC int16_t add(int16_t a, int16_t b) { return a + b; }
  OF_NUMERICS_FUNC int16_t mul(int16_t a, int16_t b) { return a * b; }
  OF_NUMERICS_FUNC int16_t sub(int16_t a, int16_t b) { return a - b; }
  OF_NUMERICS_FUNC int16_t div(int16_t a, int16_t b) { return a / b; }
  OF_NUMERICS_FUNC int16_t pow(int16_t a, int16_t b) { return powi<int16_t>(a, b); }
  OF_NUMERICS_FUNC bool isnan(int16_t a) { return false; }
  OF_NUMERICS_FUNC bool isinf(int16_t a) { return false; }
};

template<>
struct numerics<int32_t> {
  OF_NUMERICS_FUNC int32_t min() { return detail::numeric_limits<int32_t>::lowest(); }
  OF_NUMERICS_FUNC int32_t max() { return detail::numeric_limits<int32_t>::max(); }
  OF_NUMERICS_FUNC int32_t lower_bound() { return detail::numeric_limits<int32_t>::lower_bound(); }
  OF_NUMERICS_FUNC int32_t upper_bound() { return detail::numeric_limits<int32_t>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(int32_t a, int32_t b) { return a < b; }
  OF_NUMERICS_FUNC bool le(int32_t a, int32_t b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(int32_t a, int32_t b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(int32_t a, int32_t b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(int32_t a, int32_t b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(int32_t a, int32_t b) { return a != b; }

  OF_NUMERICS_FUNC int32_t add(int32_t a, int32_t b) { return a + b; }
  OF_NUMERICS_FUNC int32_t mul(int32_t a, int32_t b) { return a * b; }
  OF_NUMERICS_FUNC int32_t sub(int32_t a, int32_t b) { return a - b; }
  OF_NUMERICS_FUNC int32_t div(int32_t a, int32_t b) { return a / b; }
  OF_NUMERICS_FUNC int32_t pow(int32_t a, int32_t b) { return powi<int32_t>(a, b); }
  OF_NUMERICS_FUNC bool isnan(int32_t a) { return false; }
  OF_NUMERICS_FUNC bool isinf(int32_t a) { return false; }
};

template<>
struct numerics<int64_t> {
  OF_NUMERICS_FUNC int64_t min() { return detail::numeric_limits<int64_t>::lowest(); }
  OF_NUMERICS_FUNC int64_t max() { return detail::numeric_limits<int64_t>::max(); }
  OF_NUMERICS_FUNC int64_t lower_bound() { return detail::numeric_limits<int64_t>::lower_bound(); }
  OF_NUMERICS_FUNC int64_t upper_bound() { return detail::numeric_limits<int64_t>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(int64_t a, int64_t b) { return a < b; }
  OF_NUMERICS_FUNC bool le(int64_t a, int64_t b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(int64_t a, int64_t b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(int64_t a, int64_t b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(int64_t a, int64_t b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(int64_t a, int64_t b) { return a != b; }

  OF_NUMERICS_FUNC int64_t add(int64_t a, int64_t b) { return a + b; }
  OF_NUMERICS_FUNC int64_t mul(int64_t a, int64_t b) { return a * b; }
  OF_NUMERICS_FUNC int64_t sub(int64_t a, int64_t b) { return a - b; }
  OF_NUMERICS_FUNC int64_t div(int64_t a, int64_t b) { return a / b; };
  OF_NUMERICS_FUNC int64_t pow(int64_t a, int64_t b) { return powi<int64_t>(a, b); }
  OF_NUMERICS_FUNC bool isnan(int64_t a) { return false; }
  OF_NUMERICS_FUNC bool isinf(int64_t a) { return false; }
};

// DEPRECATED: use math functions from std and cuda math API (if needed)
template<>
struct numerics<float> {
  OF_NUMERICS_FUNC float min() { return detail::numeric_limits<float>::lowest(); }
  OF_NUMERICS_FUNC float max() { return detail::numeric_limits<float>::max(); }
  OF_NUMERICS_FUNC float lower_bound() { return detail::numeric_limits<float>::lower_bound(); }
  OF_NUMERICS_FUNC float upper_bound() { return detail::numeric_limits<float>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(float a, float b) { return a < b; }
  OF_NUMERICS_FUNC bool le(float a, float b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(float a, float b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(float a, float b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(float a, float b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(float a, float b) { return a != b; }

  OF_NUMERICS_FUNC float sqrt(float a) { return sqrtf(a); }
  OF_NUMERICS_FUNC float atan(float a) { return atanf(a); }
  OF_NUMERICS_FUNC float add(float a, float b) { return a + b; }
  OF_NUMERICS_FUNC float div(float a, float b) { return a / b; }
  OF_NUMERICS_FUNC float mul(float a, float b) { return a * b; }
  OF_NUMERICS_FUNC float sub(float a, float b) { return a - b; }
  OF_NUMERICS_FUNC float pow(float a, float b) { return powf(a, b); }
  OF_NUMERICS_FUNC bool isnan(float a) { return ::isnan(a); }
  OF_NUMERICS_FUNC bool isinf(float a) { return ::isinf(a); }
};

#if defined(__CUDACC__)
template<>
struct numerics<half> {
  OF_NUMERICS_FUNC bool isnan(half a) { return ::isnan((float)a); }
};
#endif

template<>
struct numerics<double> {
  OF_NUMERICS_FUNC double min() { return detail::numeric_limits<double>::lowest(); }
  OF_NUMERICS_FUNC double max() { return detail::numeric_limits<double>::max(); }
  OF_NUMERICS_FUNC double lower_bound() { return detail::numeric_limits<double>::lower_bound(); }
  OF_NUMERICS_FUNC double upper_bound() { return detail::numeric_limits<double>::upper_bound(); }

  OF_NUMERICS_FUNC bool lt(double a, double b) { return a < b; }
  OF_NUMERICS_FUNC bool le(double a, double b) { return a <= b; }
  OF_NUMERICS_FUNC bool gt(double a, double b) { return a > b; }
  OF_NUMERICS_FUNC bool ge(double a, double b) { return a >= b; }
  OF_NUMERICS_FUNC bool eq(double a, double b) { return a == b; }
  OF_NUMERICS_FUNC bool ne(double a, double b) { return a != b; }

  OF_NUMERICS_FUNC double sqrt(double a) { return ::sqrt(a); }
  OF_NUMERICS_FUNC double atan(double a) { return ::atan(a); }
  OF_NUMERICS_FUNC double add(double a, double b) { return a + b; }
  OF_NUMERICS_FUNC double div(double a, double b) { return a / b; }
  OF_NUMERICS_FUNC double mul(double a, double b) { return a * b; }
  OF_NUMERICS_FUNC double sub(double a, double b) { return a - b; }
  OF_NUMERICS_FUNC double pow(double a, double b) { return ::pow(a, b); }
  OF_NUMERICS_FUNC bool isnan(double a) { return ::isnan(a); }
  OF_NUMERICS_FUNC bool isinf(double a) { return ::isinf(a); }
};

}  // namespace detail
}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_NUMERICS_H
