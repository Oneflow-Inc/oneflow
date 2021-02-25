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
template<template<typename, typename> class Opt, typename PredT, typename LabelT>
struct ElemwiseSigmoidCrossEntropyGradFunctor<DeviceType::kCPU, Opt, PredT, LabelT> final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* prediction_diff, const PredT* prediction,
                  const LabelT* label, const PredT* loss_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      prediction_diff[i] = Opt<PredT, LabelT>()(prediction[i], label[i], loss_diff[i]);
    }
  }
};

template<template<typename, typename> class Opt, typename PredT, typename LabelT>
struct ElemwiseSigmoidCrossEntropyFunctor<DeviceType::kCPU, Opt, PredT, LabelT> final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* loss, const PredT* prediction,
                  const LabelT* label) {
    FOR_RANGE(int64_t, i, 0, n) { loss[i] = Opt<PredT, LabelT>()(prediction[i], label[i]); }
  }
};
}  // namespace

REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, float, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, double, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, float, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, double, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, float, float)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kCPU, double, double)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, float, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, double, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, float, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, double, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, float, float)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kCPU, double, double)

}  // namespace oneflow
