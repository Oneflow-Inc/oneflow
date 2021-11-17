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
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void PReluForwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                const int32_t inner_size, const T* x, const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T alpha_i = alpha[(i / inner_size) % alpha_size];
    y[i] = x_i > 0 ? x_i : x_i * alpha_i;
  }
}

template<typename T>
__global__ void PReluBackwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                 const int32_t inner_size, const T* x, const T* alpha, const T* dy,
                                 T* dx, T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    const T alpha_i = alpha[(i / inner_size) % alpha_size];
    dx[i] = x_i > 0 ? dy_i : dy_i * alpha_i;
    alpha_diff[(i / inner_size) % alpha_size] += x_i > 0 ? 0 : dy_i * x_i;
  }
}

}  // namespace

template<typename T>
class GpuPReluKernel final : public user_op::OpKernel {
 public:
  GpuPReluKernel() = default;
  ~GpuPReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int batch = x->shape().At(0);
    const int channels = x->shape().At(1);
    const int32_t inner_size = elem_cnt / batch / channels;
    PReluForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_PRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("prelu").SetCreateFn<GpuPReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kGPU)                                  \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_PRELU_KERNEL(float)
REGISTER_GPU_PRELU_KERNEL(double)

template<typename T>
class GpuPReluGradKernel final : public user_op::OpKernel {
 public:
  GpuPReluGradKernel() = default;
  ~GpuPReluGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);

    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int batch = x->shape().At(0);
    const int channels = x->shape().At(1);
    const int32_t inner_size = elem_cnt / batch / channels;

    Memset<DeviceType::kGPU>(ctx->device_ctx(), alpha_diff->mut_dptr<T>(), 0,
                             alpha_diff->shape().elem_cnt() * sizeof(T));

    PReluBackwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
        dx->mut_dptr<T>(), alpha_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_PRELU_GRAD_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("prelu_grad")                                  \
      .SetCreateFn<GpuPReluGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_PRELU_GRAD_KERNEL(float)
REGISTER_GPU_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
