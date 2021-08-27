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

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/diagonal_kernel.h"

namespace oneflow {
namespace {

template<typename T>
__global__ void forward_diagonal_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t dim1,
                                        int32_t dim2) {
  CUDA_1D_KERNEL_LOOP(index, size * dim2) {
    int32_t i = index / dim2;
    int32_t j = index % dim2;
    out_buf[j * size + i] = in_buf[i * (dim1 + 1) * dim2 + j];
  }
}

template<typename T>
__global__ void backward_diagonal_kernel(T* dx_buf, const T* dy_buf, int32_t size, int32_t dim1,
                                         int32_t dim2) {
  CUDA_1D_KERNEL_LOOP(index, size * dim2) {
    int32_t i = index / dim2;
    int32_t j = index % dim2;
    dx_buf[i * (dim1 + 1) * dim2 + j] = dy_buf[j * size + i];
  }
}

template<typename T>
struct DiagonalFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    if (size * dim2 > 0) {
      forward_diagonal_kernel<<<BlocksNum4ThreadsNum(size * dim2), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(out_buf, in_buf, size, dim1, dim2);
    }
  }
};

template<typename T>
struct DiagonalGradFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    if (size * dim2 > 0) {
      backward_diagonal_kernel<<<BlocksNum4ThreadsNum(size * dim2), kCudaThreadsNumPerBlock, 0,
                                 ctx->cuda_stream()>>>(dx_buf, dy_buf, size, dim1, dim2);
    }
  }
};

}  // namespace

REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, half);
REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, float);
REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, double);
REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, int8_t);
REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, int32_t);
REGISTER_DIAGONAL_KERNELS(DeviceType::kGPU, int64_t);

}  // namespace oneflow