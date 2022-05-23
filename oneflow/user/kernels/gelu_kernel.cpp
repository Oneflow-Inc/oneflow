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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::ElementwiseUnary> NewGeluPrimitive(Context* ctx) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kGelu, src->data_type(), dst->data_type());
}

class GeluKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluKernel);
  GeluKernel() = default;
  ~GeluKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto primitive = NewGeluPrimitive(ctx);
    CHECK(primitive);

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = x->shape().elem_cnt();

    if (elem_cnt != 0) {
      primitive->Launch(ctx->stream(), x->dptr(), y->mut_dptr(), elem_cnt);
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto GeluPrimitiveExists() {
  return hob::make_custom("GeluPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewGeluPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("gelu").SetCreateFn<GeluKernel>().SetIsMatchedHob(GeluPrimitiveExists()
                                                                       == true);

template<typename T>
class CpuGeluGradKernel final : public user_op::OpKernel {
 public:
  CpuGeluGradKernel() = default;
  ~CpuGeluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    T inv_sqrt2 = std::sqrt(0.5);
    T coef = std::sqrt(2.0 / std::acos(-1.0));
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = 0.5
                  * (1.0 + std::erf(inv_sqrt2 * x_ptr[i])
                     + x_ptr[i] * coef * std::exp(-0.5 * x_ptr[i] * x_ptr[i]))
                  * dy_ptr[i];
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_GELU_GRAD_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("gelu_grad")                                   \
      .SetCreateFn<CpuGeluGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_GELU_GRAD_KERNEL(float)
REGISTER_CPU_GELU_GRAD_KERNEL(double)

}  // namespace oneflow
