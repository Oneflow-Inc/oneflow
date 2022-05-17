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

template<typename T>
class CpuPReluKernel final : public user_op::OpKernel {
 public:
  CpuPReluKernel() = default;
  ~CpuPReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x_ptr = x->dptr<T>();
    const T* alpha_ptr = alpha->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int batch = x->shape().At(0);
    const int channels = (x->shape().NumAxes() == 1) ? 1 : x->shape().At(1);
    const int32_t inner_size = elem_cnt / batch / channels;
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : x_ptr[i] * alpha_ptr[(i / inner_size) % alpha_size];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_PRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("prelu").SetCreateFn<CpuPReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                  \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_PRELU_KERNEL(float)
REGISTER_CPU_PRELU_KERNEL(double)

template<typename T>
class CpuPReluGradKernel final : public user_op::OpKernel {
 public:
  CpuPReluGradKernel() = default;
  ~CpuPReluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    const T* x_ptr = x->dptr<T>();
    const T* alpha_ptr = alpha->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    T* alpha_diff_ptr = alpha_diff->mut_dptr<T>();

    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int batch = x->shape().At(0);
    const int channels = (x->shape().NumAxes() == 1) ? 1 : x->shape().At(1);
    const int32_t inner_size = elem_cnt / batch / channels;

    Memset<DeviceType::kCPU>(ctx->stream(), alpha_diff->mut_dptr<T>(), 0,
                             alpha_diff->shape().elem_cnt() * sizeof(T));

    for (int i = 0; i < elem_cnt; i++) {
      const T x_i = x_ptr[i];
      const T dy_i = dy_ptr[i];
      const T alpha_i = alpha_ptr[(i / inner_size) % alpha_size];
      dx_ptr[i] = x_i > 0 ? dy_i : dy_i * alpha_i;
      alpha_diff_ptr[(i / inner_size) % alpha_size] += x_i > 0 ? 0 : dy_i * x_i;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_PRELU_GRAD_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("prelu_grad")                                  \
      .SetCreateFn<CpuPReluGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_PRELU_GRAD_KERNEL(float)
REGISTER_CPU_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
