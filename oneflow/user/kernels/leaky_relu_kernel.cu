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

namespace oneflow {

namespace {

template<typename T>
__global__ void LeakyReluForwardGpu(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha; }
}

template<typename T>
__global__ void LeakyReluBackwardGpu(const int n, const float alpha, const T* x, const T* dy,
                                     T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha; }
}

}  // namespace

template<typename T>
class GpuLeakyReluKernel final : public user_op::OpKernel {
 public:
  GpuLeakyReluKernel() = default;
  ~GpuLeakyReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->Attr<float>("alpha");
    RUN_CUDA_KERNEL((LeakyReluForwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, alpha,
                    x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_LEAKY_RELU_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("leaky_relu")                      \
      .SetCreateFn<GpuLeakyReluKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_LEAKY_RELU_KERNEL(float)
REGISTER_GPU_LEAKY_RELU_KERNEL(double)

template<typename T>
class GpuLeakyReluGradKernel final : public user_op::OpKernel {
 public:
  GpuLeakyReluGradKernel() = default;
  ~GpuLeakyReluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->Attr<float>("alpha");
    RUN_CUDA_KERNEL((LeakyReluBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, alpha,
                    x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("leaky_relu_grad")                 \
      .SetCreateFn<GpuLeakyReluGradKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(float)
REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(double)

}  // namespace oneflow
