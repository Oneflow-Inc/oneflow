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
#ifndef ONEFLOW_CORE_VECTORIZED_VEC256_DOUBLE_H_
#define ONEFLOW_CORE_VECTORIZED_VEC256_DOUBLE_H_
#include "oneflow/core/vectorized/vec256/vec256_base.h"

namespace oneflow {

#ifdef WITH_AVX
#include <immintrin.h>

template<>
class VectorizedAvx2<double> {
 public:
  static void fmadd(size_t begin, size_t end, const double* x, const double* y, double* out, double alpha) {
    size_t i = begin;
    size_t stride = 4;

    __m256d _alpha = _mm256_set1_pd(alpha);

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256d _x1 = _mm256_loadu_pd(x + i);
      __m256d _x2 = _mm256_loadu_pd(x + i + stride);

      __m256d _y1 = _mm256_loadu_pd(y + i);
      __m256d _y2 = _mm256_loadu_pd(y + i + stride);

      __m256d _o1 = _mm256_fmadd_pd(_x1, _alpha, _y1);
      __m256d _o2 = _mm256_fmadd_pd(_x2, _alpha, _y2);

      _mm256_storeu_pd(out + i, _o1);
      _mm256_storeu_pd(out + i + stride, _o2);
    }

    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] * alpha + y[i]; }
    }
  }

  static void add(size_t begin, size_t end, const double* x, const double* y, double* out) {
    size_t i = begin;
    size_t stride = 4;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256d _x1 = _mm256_loadu_pd(x + i);
      __m256d _x2 = _mm256_loadu_pd(x + i + stride);

      __m256d _y1 = _mm256_loadu_pd(y + i);
      __m256d _y2 = _mm256_loadu_pd(y + i + stride);

      __m256d _o1 = _mm256_add_pd(_x1, _y1);
      __m256d _o2 = _mm256_add_pd(_x2, _y2);

      _mm256_storeu_pd(out + i, _o1);
      _mm256_storeu_pd(out + i + stride, _o2);
    }

    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] + y[i]; }
    }
  }

  static void sub(size_t begin, size_t end, const double* x, const double* y, double* out) {
    size_t i = begin;
    size_t stride = 4;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256d _x1 = _mm256_loadu_pd(x + i);
      __m256d _x2 = _mm256_loadu_pd(x + i + stride);

      __m256d _y1 = _mm256_loadu_pd(y + i);
      __m256d _y2 = _mm256_loadu_pd(y + i + stride);

      __m256d _o1 = _mm256_sub_pd(_x1, _y1);
      __m256d _o2 = _mm256_sub_pd(_x2, _y2);

      _mm256_storeu_pd(out + i, _o1);
      _mm256_storeu_pd(out + i + stride, _o2);
    }

    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] - y[i]; }
    }
  }

  static void mul(size_t begin, size_t end, const double* x, const double* y, double* out) {
    size_t i = begin;
    size_t stride = 4;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256d _x1 = _mm256_loadu_pd(x + i);
      __m256d _x2 = _mm256_loadu_pd(x + i + stride);

      __m256d _y1 = _mm256_loadu_pd(y + i);
      __m256d _y2 = _mm256_loadu_pd(y + i + stride);

      __m256d _o1 = _mm256_mul_pd(_x1, _y1);
      __m256d _o2 = _mm256_mul_pd(_x2, _y2);

      _mm256_storeu_pd(out + i, _o1);
      _mm256_storeu_pd(out + i + stride, _o2);
    }

    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] * y[i]; }
    }
  }

  static void div(size_t begin, size_t end, const double* x, const double* y, double* out) {
    size_t i = begin;
    size_t stride = 4;

    for (; i <= end - 2 * stride; i += 2 * stride) {
      __m256d _x1 = _mm256_loadu_pd(x + i);
      __m256d _x2 = _mm256_loadu_pd(x + i + stride);

      __m256d _y1 = _mm256_loadu_pd(y + i);
      __m256d _y2 = _mm256_loadu_pd(y + i + stride);

      __m256d _o1 = _mm256_div_pd(_x1, _y1);
      __m256d _o2 = _mm256_div_pd(_x2, _y2);

      _mm256_storeu_pd(out + i, _o1);
      _mm256_storeu_pd(out + i + stride, _o2);
    }

    if (i < end) {
      for (; i < end; i++) { out[i] = x[i] / y[i]; }
    }
  }
};

#endif

}  // namespace oneflow

#endif