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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/sigmoid_cross_entropy_kernel.h"

namespace oneflow {

namespace {
template<template<typename, typename> class Opt, typename PredT, typename LabelT>
struct ElemwiseSigmoidCrossEntropyGradFunctor<DeviceType::kGPU, Opt, PredT, LabelT> final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* prediction_diff, const PredT* prediction,
                  const LabelT* label, const PredT* loss_diff) {
    OF_CUDA_CHECK(cuda::elementwise::Ternary(Opt<PredT, LabelT>(), n, prediction_diff, prediction,
                                             label, loss_diff, ctx->cuda_stream()));
  }
};

template<template<typename, typename> class Opt, typename PredT, typename LabelT>
struct ElemwiseSigmoidCrossEntropyFunctor<DeviceType::kGPU, Opt, PredT, LabelT> final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* loss, const PredT* prediction,
                  const LabelT* label) {
    OF_CUDA_CHECK(cuda::elementwise::Binary(Opt<PredT, LabelT>(), n, loss, prediction, label,
                                            ctx->cuda_stream()));
  }
};
}  // namespace
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, float, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, double, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, float, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, double, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, float, float)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, double, double)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, float, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, double, int32_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, float, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, double, int8_t)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, float, float)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, double, double)

}  // namespace oneflow
