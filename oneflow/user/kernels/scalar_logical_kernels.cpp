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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewBinaryPrimitive(
    Context* ctx, ep::primitive::BinaryOp op) {
  const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  const int64_t ndims = in->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, in->data_type(), out->data_type(), ndims);
}

template<ep::primitive::BinaryOp op>
auto PrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseBinaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBinaryPrimitive(&ctx, op).operator bool();
                          });
}

template<ep::primitive::BinaryOp op>
class ScalarLogicalKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ScalarLogicalKernel() = default;
  ~ScalarLogicalKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar scalar_operand;
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = ctx->Attr<int64_t>("int_operand");
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = ctx->Attr<double>("float_operand");
    } else {
      UNIMPLEMENTED();
    }

    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), in->shape_view().NumAxes(), in->shape_view().ptr(),
                        in->dptr(), scalar_operand, out->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(kernel_name, binary_op) \
  REGISTER_USER_KERNEL(kernel_name)                                                \
      .SetCreateFn<ScalarLogicalKernel<binary_op>>()                               \
      .SetIsMatchedHob(PrimitiveExists<binary_op>());

REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_equal",
                                                   ep::primitive::BinaryOp::kEqual);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_not_equal",
                                                   ep::primitive::BinaryOp::kNotEqual);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_greater",
                                                   ep::primitive::BinaryOp::kGreaterThan);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_greater_equal",
                                                   ep::primitive::BinaryOp::kGreaterEqual);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_less",
                                                   ep::primitive::BinaryOp::kLessThan);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_less_equal",
                                                   ep::primitive::BinaryOp::kLessEqual);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_or",
                                                   ep::primitive::BinaryOp::kLogicalOr);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_xor",
                                                   ep::primitive::BinaryOp::kLogicalXor);
REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL("scalar_logical_and",
                                                   ep::primitive::BinaryOp::kLogicalAnd);

}  // namespace

}  // namespace oneflow
