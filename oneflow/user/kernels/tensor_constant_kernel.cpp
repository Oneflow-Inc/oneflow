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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/tensor_fill.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::TensorFill> NewTensorFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::TensorFillFactory>(ctx->device_type(),
                                                                       data_type);
}

class TensorConstantKernel final : public OpKernel {
 public:
  TensorConstantKernel() = default;
  ~TensorConstantKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* value_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK(value_tensor->shape_view().NumAxes() <= 1 && value_tensor->shape_view().elem_cnt() == 1)
        << "Only scalar tensor as filled value is supported!";

    const int64_t elem_cnt = out_tensor->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }
    std::unique_ptr<ep::primitive::TensorFill> tensor_fill = NewTensorFillPrimitive(ctx);
    CHECK(tensor_fill);
    tensor_fill->Launch(ctx->stream(), value_tensor->raw_dptr(), out_tensor->mut_dptr(), elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto TensorFillPrimitiveExists() {
  return hob::make_custom("TensorFillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewTensorFillPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("tensor_constant")
    .SetCreateFn<TensorConstantKernel>()
    .SetIsMatchedHob(TensorFillPrimitiveExists() == true);

}  // namespace

}  // namespace user_op
}  // namespace oneflow
