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
#include "oneflow/user/kernels/sigmoid_cross_entropy_kernel.h"

namespace oneflow {

namespace {
template<template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyGradFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, int64_t n, T* prediction_diff, const T* prediction,
                  const T* label) {
    XPU_1D_KERNEL_LOOP(index, n) {
      prediction_diff[index] = Opt<T>()(prediction[index], label[index]);
    }
  }
};

template<template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyFunctor<DeviceType::kCPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, int64_t n, T* loss, const T* prediction, const T* label) {
    XPU_1D_KERNEL_LOOP(index, n) { loss[index] = Opt<T>()(prediction[index], label[index]); }
  }
};
}  // namespace

REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, float)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, double)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, double)

}  // namespace oneflow
