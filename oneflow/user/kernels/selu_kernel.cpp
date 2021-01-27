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
#include "oneflow/user/kernels/selu_kernel.h"

namespace oneflow {

namespace {

template<template<typename> class Opt, typename T>
struct ElemwiseSeluFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T scale, T alpha, T* out, const T* in) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = Opt<T>(scale, alpha)(in[i]); }
  }
};

template<template<typename> class Opt, typename T>
struct ElemwiseSeluGradFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T scale, T alpha, T* dx, const T* y,
                  const T* dy) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { dx[i] = Opt<T>(scale, alpha)(y[i], dy[i]); }
  }
};

}  // namespace

REGISTER_SELU_KERNELS(DeviceType::kCPU, float);
REGISTER_SELU_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow