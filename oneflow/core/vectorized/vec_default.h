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
#ifndef ONEFLOW_CORE_VECTORIZED_VEC_DEFAULT_H_
#define ONEFLOW_CORE_VECTORIZED_VEC_DEFAULT_H_

#include <iostream>

namespace oneflow {

template<typename T>
class VectorizedDefault {
 public:
  static void fmadd(size_t begin, size_t end, const T* x, const T* y, T* out, const T alpha) {
    for (size_t i = begin; i <= end; i++) { out[i] = x[i] * alpha + y[i]; }
  }

  static void add(size_t begin, size_t end, const T* x, const T* y, T* out) {
    for (size_t i = begin; i <= end; i++) { out[i] = x[i] + y[i]; }
  }

  static void sub(size_t begin, size_t end, const T* x, const T* y, T* out) {
    for (size_t i = begin; i <= end; i++) { out[i] = x[i] - y[i]; }
  }

  static void mul(size_t begin, size_t end, const T* x, const T* y, T* out) {
    for (size_t i = begin; i <= end; i++) { out[i] = x[i] * y[i]; }
  }

  static void div(size_t begin, size_t end, const T* x, const T* y, T* out) {
    for (size_t i = begin; i <= end; i++) { out[i] = x[i] / y[i]; }
  }
};

}  // namespace oneflow

#endif
