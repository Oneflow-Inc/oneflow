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
#include "oneflow/user/kernels/hardsigmoid_kernel.h"

namespace oneflow {

namespace {

template<template<typename> class Opt, typename T>
struct ElemwiseHardsigmoidFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T* out, const T* in) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      if (in[i] <= static_cast<T>(-3)) {
        out[i] = static_cast<T>(0);
      } else if (in[i] >= static_cast<T>(3)) {
        out[i] = static_cast<T>(1);
      } else {
        out[i] = (in[i] / static_cast<T>(6)) + static_cast<T>(0.5);
      }
    }
  }
};

template<template<typename> class Opt, typename T>
struct ElemwiseHardsigmoidGradFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T* dx, const T* x, const T* dy) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      dx[i] = (x[i] > static_cast<T>(-3) && x[i] < static_cast<T>(3)) ? dy[i] / static_cast<T>(6)
                                                                      : static_cast<T>(0);
    }
  }
};

}  // namespace

REGISTER_HARDSIGMOID_KERNELS(DeviceType::kCPU, float);
REGISTER_HARDSIGMOID_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow
