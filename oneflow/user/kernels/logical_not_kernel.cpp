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
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::ElementwiseUnary> NewLogicalNotPrimitive(Context* ctx) {
  const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kLogicalNot, in_data_type, out_data_type);
}

class LogicalNotKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LogicalNotKernel() = default;
  ~LogicalNotKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t n = tensor_x->shape_view().elem_cnt();

    if (n != 0) {
      auto primitive = NewLogicalNotPrimitive(ctx);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), tensor_x->dptr(), tensor_y->mut_dptr(), n);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto LogicalNotPrimitiveExists() {
  return hob::make_custom("LogicalNotPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) -> bool {
                            return NewLogicalNotPrimitive(&ctx).operator bool();
                          });
}

REGISTER_USER_KERNEL("logical_not")
    .SetCreateFn<LogicalNotKernel>()
    .SetIsMatchedHob(LogicalNotPrimitiveExists());
}  // namespace

}  // namespace oneflow
