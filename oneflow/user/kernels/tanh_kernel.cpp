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
#include <cmath>

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::ElementwiseUnary> NewTanhPrimitive(Context* ctx) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kTanh, src->data_type(), dst->data_type());
}

class TanhKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TanhKernel);
  TanhKernel() = default;
  ~TanhKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto primitive = NewTanhPrimitive(ctx);
    CHECK(primitive);

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
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

auto TanhPrimitiveExists() {
  return hob::make_custom("TanhPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewTanhPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("tanh").SetCreateFn<TanhKernel>().SetIsMatchedHob(TanhPrimitiveExists()
                                                                       == true);

template<typename T>
class CpuTanhGradKernel final : public user_op::OpKernel {
 public:
  CpuTanhGradKernel() = default;
  ~CpuTanhGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      T tanh_val = std::tanh(x_ptr[i]);
      dx_ptr[i] = dy_ptr[i] * (static_cast<T>(1.0) - tanh_val * tanh_val);
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_TANH_GRAD_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL((std::string("") + "tanh" + "_grad"))          \
      .SetCreateFn<CpuTanhGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_TANH_GRAD_KERNEL(float)
REGISTER_CPU_TANH_GRAD_KERNEL(double)

}  // namespace oneflow
