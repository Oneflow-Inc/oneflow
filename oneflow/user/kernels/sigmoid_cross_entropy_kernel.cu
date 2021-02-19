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
#ifdef WITH_CUDA
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/sigmoid_cross_entropy_kernel.h"

namespace oneflow {

namespace {
template<template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyGradFunctor<DeviceType::kGPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, int64_t n, T* prediction_diff, const T* prediction,
                  const T* label) {
    OF_CUDA_CHECK(cuda::elementwise::Binary(Opt<T>(), n, prediction_diff, prediction, label,
                                            ctx->cuda_stream()));
  }
};

template<template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyFunctor<DeviceType::kGPU, Opt, T> final {
  void operator()(DeviceCtx* ctx, int64_t n, T* loss, const T* prediction, const T* label) {
    OF_CUDA_CHECK(
        cuda::elementwise::Binary(Opt<T>(), n, loss, prediction, label, ctx->cuda_stream()));
  }
};
}  // namespace

REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, float)
REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(DeviceType::kGPU, double)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(DeviceType::kGPU, double)

}  // namespace oneflow
#endif  // WITH_CUDA
