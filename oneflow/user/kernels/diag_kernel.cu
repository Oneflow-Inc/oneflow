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

namespace oneflow {
namespace {

template<typename T>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void compute_diag_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t strideSum,
                                    int32_t in_dim) {
  if (in_dim == 1) {
    CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i * (strideSum)] = in_buf[i]; }
  } else {
    CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i] = in_buf[i * (strideSum)]; }
  }
}

template<typename T>
__global__ void compute_diag_grad_kernel(T* out_buf, const T* in_buf, int32_t size,
                                         int32_t strideSum, int32_t in_dim) {
  if (in_dim != 1) {
    CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i * (strideSum)] = in_buf[i]; }
  } else {
    CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i] = in_buf[i * (strideSum)]; }
  }
}

template<typename T>
class DiagKernelGPU final : public user_op::OpKernel {
 public:
  DiagKernelGPU() = default;
  ~DiagKernelGPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& out_shape = out->shape();
    const ShapeView& in_shape = in->shape();
    int32_t in_dim = in_shape.NumAxes();

    Memset<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr(), 0,
                             out_shape.elem_cnt() * sizeof(T));

    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();

    if (in_dim == 1) {
      int32_t stride_0 = out_shape.At(1);
      int32_t stride_1 = 1;
      int32_t input_cnt = in_shape.elem_cnt();

      out_buf += (dimension >= 0 ? dimension * stride_1 : -dimension * stride_0);
      // Kernel Launch
      compute_diag_kernel<<<BlocksNum4ThreadsNum(input_cnt * input_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
          out_buf, in_buf,

          input_cnt, int32_t(stride_0 + stride_1), in_dim);
    } else {
      int32_t stride_0 = in_shape.At(1);
      int32_t stride_1 = 1;
      int32_t sz = 0;

      in_buf += (dimension >= 0 ? dimension * stride_1 : -dimension * stride_0);
      if (dimension >= 0) {
        sz = std::min(in_shape.At(0), in_shape.At(1) - dimension);
      } else {
        sz = std::min(in_shape.At(0) + dimension, in_shape.At(1));
      }

      // Kernel Launch
      compute_diag_kernel<<<BlocksNum4ThreadsNum(sz * sz), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
          out_buf, in_buf, sz, int32_t(stride_0 + stride_1), in_dim);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class DiagGradKernelGPU final : public user_op::OpKernel {
 public:
  DiagGradKernelGPU() = default;
  ~DiagGradKernelGPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int32_t dimension = ctx->Attr<int32_t>("dimension");
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();
    int32_t in_dim = dx_shape.NumAxes();
    int32_t dy_num_cnt = dy_shape.At(0);
    int32_t dx_num_cnt = dx_shape.Count(0);
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();

    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx->mut_dptr<T>(), 0,
                             dx_shape.elem_cnt() * sizeof(T));

    if (in_dim == 1) {
      int32_t stride_1 = 1;
      int32_t stride_0 = dy_shape.At(1);

      dy_buf += (dimension >= 0 ? dimension * stride_1 : -dimension * stride_0);
      compute_diag_grad_kernel<<<BlocksNum4ThreadsNum(dx_num_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
          dx_buf, dy_buf, dx_num_cnt, int32_t(stride_0 + stride_1), in_dim);
    } else {
      int32_t stride_0 = dx_shape.At(1);
      int32_t stride_1 = 1;
      dx_buf += (dimension >= 0 ? dimension * stride_1 : -dimension * stride_0);
      compute_diag_grad_kernel<<<BlocksNum4ThreadsNum(dy_num_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
          dx_buf, dy_buf, dy_num_cnt, int32_t(stride_0 + stride_1), in_dim);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIAG_KERNEL_GPU(dtype)                                             \
  REGISTER_USER_KERNEL("diag").SetCreateFn<DiagKernelGPU<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                            \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_DIAG_KERNEL_GPU(half)
REGISTER_DIAG_KERNEL_GPU(float)
REGISTER_DIAG_KERNEL_GPU(double)

#define REGISTER_DIAG_GRAD_KERNEL_GPU(dtype)              \
  REGISTER_USER_KERNEL("diag_grad")                       \
      .SetCreateFn<DiagGradKernelGPU<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_DIAG_GRAD_KERNEL_GPU(half)
REGISTER_DIAG_GRAD_KERNEL_GPU(float)
REGISTER_DIAG_GRAD_KERNEL_GPU(double)

}  // namespace
}  // namespace oneflow