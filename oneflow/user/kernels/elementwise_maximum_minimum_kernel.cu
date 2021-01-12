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

template<template<typename> class XmumUtil, typename T>
__global__ void ElementwiseBackwardGradGpu(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                           T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    XmumUtil<T>::Backward(&dz[idx], &x[idx], &y[idx], dx ? &dx[idx] : nullptr,
                          dy ? &dy[idx] : nullptr);
  }
}

template<template<typename> class XumUtil, typename T>
struct RunKernelUtil<DeviceType::kGPU, XumUtil, T> final {
  static void BackwardKernel(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y,
                             T* dx, T* dy) {
    ElementwiseBackwardGradGpu<XumUtil, T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, dz, x, y, dx, dy);
  }

  static void ForwardKernel(DeviceCtx* ctx, int64_t elem_cnt, T* z, const T* x, const T* y) {
    OF_CUDA_CHECK(cuda::elementwise::Binary(XumUtil<T>(), elem_cnt, z, x, y, ctx->cuda_stream()));
  }
};

#define REGISTER_XMUM_GPU_KERNELS(op_type_name, util, dtype)            \
  REGISTER_FORWARD_KERNEL(DeviceType::kGPU, op_type_name, util, dtype); \
  REGISTER_BACKWARD_KERNEL(DeviceType::kGPU, op_type_name, util, dtype);

REGISTER_XMUM_GPU_KERNELS("elementwise_maximum", MaximumUtil, float);
REGISTER_XMUM_GPU_KERNELS("elementwise_maximum", MaximumUtil, double);
REGISTER_XMUM_GPU_KERNELS("elementwise_minimum", MinimumUtil, float);
REGISTER_XMUM_GPU_KERNELS("elementwise_minimum", MinimumUtil, double);

}  // namespace user_op
}  // namespace oneflow
#endif  // WITH_CUDA
