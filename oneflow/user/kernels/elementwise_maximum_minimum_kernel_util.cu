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
#include "oneflow/user/kernels/elementwise_maximum_minimum_kernel_util.h"

namespace oneflow {
namespace user_op {

template<template<typename> class functor, typename T>
__global__ void ElementwiseBackwardGradGpu(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                           T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    functor<T>()(&dz[idx], &x[idx], &y[idx], dx ? &dx[idx] : nullptr, dy ? &dy[idx] : nullptr);
  }
}

template<template<typename> class functor, typename T>
struct ElemwiseXimumBackwardFunctor<DeviceType::kGPU, functor, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy) {
    ElementwiseBackwardGradGpu<functor, T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, dz, x, y, dx, dy);
  }
};

template<template<typename> class functor, typename T>
struct ElemwiseXimumForwardFunctor<DeviceType::kGPU, functor, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* z, const T* x, const T* y) {
    OF_CUDA_CHECK(cuda::elementwise::Binary(functor<T>(), elem_cnt, z, x, y, ctx->cuda_stream()));
  }
};

INSTANTIATE_FORWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MaximumForwardFunctor, float);
INSTANTIATE_FORWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MaximumForwardFunctor, double);
INSTANTIATE_FORWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MinimumForwardFunctor, float);
INSTANTIATE_FORWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MinimumForwardFunctor, double);

INSTANTIATE_BACKWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MaximumBackwardFunctor, float);
INSTANTIATE_BACKWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MaximumBackwardFunctor, double);
INSTANTIATE_BACKWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MinimumBackwardFunctor, float);
INSTANTIATE_BACKWARD_KERNEL_FUNCTOR(DeviceType::kGPU, MinimumBackwardFunctor, double);
}  // namespace user_op
}  // namespace oneflow
#endif  // WITH_CUDA
