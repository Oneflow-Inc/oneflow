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
std::unique_ptr<ep::primitive::ElementwiseUnary> NewReluPrimitive(Context* ctx) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kRelu, src->data_type(), dst->data_type());
}

class ReluKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernel);
  ReluKernel() = default;
  ~ReluKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto primitive = NewReluPrimitive(ctx);
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

auto ReluPrimitiveExists() {
  return hob::make_custom("ReluPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewReluPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("relu").SetCreateFn<ReluKernel>().SetIsMatchedHob(ReluPrimitiveExists()
                                                                       == true);

}  // namespace oneflow
