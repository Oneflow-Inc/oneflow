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

template<typename T>
class CpuLeakyReluKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluKernel() = default;
  ~CpuLeakyReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->Attr<float>("alpha");
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : x_ptr[i] * alpha; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LEAKY_RELU_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("leaky_relu")                      \
      .SetCreateFn<CpuLeakyReluKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LEAKY_RELU_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_KERNEL(double)

template<typename T>
class CpuLeakyReluGradKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluGradKernel() = default;
  ~CpuLeakyReluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->Attr<float>("alpha");
    const T* x_ptr = x->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { dx_ptr[i] = x_ptr[i] > 0 ? dy_ptr[i] : dy_ptr[i] * alpha; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("leaky_relu_grad")                 \
      .SetCreateFn<CpuLeakyReluGradKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(double)

}  // namespace oneflow
