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
#ifndef ONEFLOW_CORE_VECTORIZED_VEC256_FLOAT_H_
#define ONEFLOW_CORE_VECTORIZED_VEC256_FLOAT_H_
#include "oneflow/core/vectorized/vec256/vec256_base.h"

namespace oneflow {

#ifdef WITH_AVX
#include <immintrin.h>

template<>
class VectorizedAvx2<float> {
 public:
  static void fmadd(size_t begin, size_t end, const float* x, const float* y, float* out, float alpha) {
    size_t i = begin;
    size_t stride = 8;

    __m256 _alpha = _mm256_set1_ps(alpha);

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256 _x1 = _mm256_loadu_ps(x + i);
      __m256 _x2 = _mm256_loadu_ps(x + i + stride);

      __m256 _y1 = _mm256_loadu_ps(y + i);
      __m256 _y2 = _mm256_loadu_ps(y + i + stride);

      __m256 _o1 = _mm256_fmadd_ps(_x1, _alpha, _y1);
      __m256 _o2 = _mm256_fmadd_ps(_x2, _alpha, _y2);

      _mm256_storeu_ps(out + i, _o1);
      _mm256_storeu_ps(out + i + stride, _o2);
    }
    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] * alpha + y[i]; }
    }
  }
  static void add(int64_t begin, int64_t end, const float* x, const float* y, float* out) {
    int64_t i = begin;
    int64_t stride = 8;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256 _x1 = _mm256_loadu_ps(x + i);
      __m256 _x2 = _mm256_loadu_ps(x + i + stride);

      __m256 _y1 = _mm256_loadu_ps(y + i);
      __m256 _y2 = _mm256_loadu_ps(y + i + stride);

      __m256 _o1 = _mm256_add_ps(_x1, _y1);
      __m256 _o2 = _mm256_add_ps(_x2, _y2);

      _mm256_storeu_ps(out + i, _o1);
      _mm256_storeu_ps(out + i + stride, _o2);
    }
    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] + y[i]; }
    }
  }

  static void sub(size_t begin, size_t end, const float* x, const float* y, float* out) {
    size_t i = begin;
    size_t stride = 8;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256 _x1 = _mm256_loadu_ps(x + i);
      __m256 _x2 = _mm256_loadu_ps(x + i + stride);

      __m256 _y1 = _mm256_loadu_ps(y + i);
      __m256 _y2 = _mm256_loadu_ps(y + i + stride);

      __m256 _o1 = _mm256_sub_ps(_x1, _y1);
      __m256 _o2 = _mm256_sub_ps(_x2, _y2);

      _mm256_storeu_ps(out + i, _o1);
      _mm256_storeu_ps(out + i + stride, _o2);
    }
    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] - y[i]; }
    }
  }

  static void mul(int64_t begin, int64_t end, const float* x, const float* y, float* out) {
    int64_t i = begin;
    int64_t stride = 8;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256 _x1 = _mm256_loadu_ps(x + i);
      __m256 _x2 = _mm256_loadu_ps(x + i + stride);

      __m256 _y1 = _mm256_loadu_ps(y + i);
      __m256 _y2 = _mm256_loadu_ps(y + i + stride);

      __m256 _o1 = _mm256_mul_ps(_x1, _y1);
      __m256 _o2 = _mm256_mul_ps(_x2, _y2);

      _mm256_storeu_ps(out + i, _o1);
      _mm256_storeu_ps(out + i + stride, _o2);
    }
    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] * y[i]; }
    }
  }

  static void div(size_t begin, size_t end, const float* x, const float* y, float* out) {
    size_t i = begin;
    size_t stride = 8;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256 _x1 = _mm256_loadu_ps(x + i);
      __m256 _x2 = _mm256_loadu_ps(x + i + stride);

      __m256 _y1 = _mm256_loadu_ps(y + i);
      __m256 _y2 = _mm256_loadu_ps(y + i + stride);

      __m256 _o1 = _mm256_div_ps(_x1, _y1);
      __m256 _o2 = _mm256_div_ps(_x2, _y2);

      _mm256_storeu_ps(out + i, _o1);
      _mm256_storeu_ps(out + i + stride, _o2);
    }
    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] / y[i]; }
    }
  }
};

#endif

}  // namespace oneflow

#endif