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
#include "oneflow/core/ep/include/primitive/where.h"

namespace oneflow {

namespace {

template<typename Context>
auto NewPrimitive(Context* ctx) -> std::unique_ptr<ep::primitive::Where> {
  const user_op::TensorDesc* cond_desc = ctx->TensorDesc4ArgNameAndIndex("condition", 0);
  const user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  return ep::primitive::NewPrimitive<ep::primitive::WhereFactory>(
      ctx->device_type(), cond_desc->data_type(), out_desc->data_type(),
      out_desc->shape().NumAxes());
}

auto PrimitiveExists() {
  return hob::make_custom("PrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewPrimitive(&ctx).operator bool();
  });
}

class WhereKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }
    auto primitive = NewPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), cond->shape_view().size(), cond->shape_view().ptr(),
                      cond->dptr(), x->shape_view().size(), x->shape_view().ptr(), x->dptr(),
                      y->shape_view().size(), y->shape_view().ptr(), y->dptr(), out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("where").SetCreateFn<WhereKernel>().SetIsMatchedHob(PrimitiveExists() == true);

}  // namespace

}  // namespace oneflow
