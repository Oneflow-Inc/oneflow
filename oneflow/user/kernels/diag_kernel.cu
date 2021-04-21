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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/diag_kernel.h"

namespace oneflow {
namespace {

template<typename T>
__global__ void vector_diagonal_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t stride) {
  CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i * stride] = in_buf[i]; }
}

template<typename T>
__global__ void matrix_diagonal_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t stride) {
  CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i] = in_buf[i * stride]; }
}

template<typename T>
struct DiagFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t stride,
                  int32_t in_dim) {
    if (in_dim == 1) {
      vector_diagonal_kernel<<<BlocksNum4ThreadsNum(size * size), kCudaThreadsNumPerBlock, 0,
                               ctx->cuda_stream()>>>(out_buf, in_buf, size, stride);
    } else {
      matrix_diagonal_kernel<<<BlocksNum4ThreadsNum(size * size), kCudaThreadsNumPerBlock, 0,
                               ctx->cuda_stream()>>>(out_buf, in_buf, size, stride);
    }
  }
};

template<typename T>
struct DiagGradFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t dx_cnt, int32_t dy_cnt,
                  int32_t stride, int32_t in_dim) {
    if (in_dim == 1) {
      matrix_diagonal_kernel<<<BlocksNum4ThreadsNum(dx_cnt), kCudaThreadsNumPerBlock, 0,
                               ctx->cuda_stream()>>>(dx_buf, dy_buf, dx_cnt, stride);
    } else {
      vector_diagonal_kernel<<<BlocksNum4ThreadsNum(dy_cnt), kCudaThreadsNumPerBlock, 0,
                               ctx->cuda_stream()>>>(dx_buf, dy_buf, dy_cnt, stride);
    }
  }
};

}  // namespace

REGISTER_DIAG_KERNELS(DeviceType::kGPU, half);
REGISTER_DIAG_KERNELS(DeviceType::kGPU, float);
REGISTER_DIAG_KERNELS(DeviceType::kGPU, double);
REGISTER_DIAG_KERNELS(DeviceType::kGPU, int8_t);
REGISTER_DIAG_KERNELS(DeviceType::kGPU, int32_t);
REGISTER_DIAG_KERNELS(DeviceType::kGPU, int64_t);

}  // namespace oneflow
