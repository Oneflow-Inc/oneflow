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
#include "oneflow/core/primitive/include/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<primitive::BroadcastElementwiseBinary> NewBroadcastElementwiseBinaryPrimitive(
    Context* ctx, primitive::BinaryOp op) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* z = ctx->TensorDesc4ArgNameAndIndex("z", 0);
  const int64_t ndims = z->shape().NumAxes();
  return primitive::NewPrimitive<primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), z->data_type(), ndims);
}

template<primitive::BinaryOp op>
hob::HobContextGetter<user_op::KernelRegContext, bool> BroadcastElementwiseBinaryPrimitiveExists() {
  return user_op::HobCtxGetter<bool>(
      "BroadcastElementwiseBinaryPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
        return NewBroadcastElementwiseBinaryPrimitive(&ctx, op).operator bool();
      });
}

}  // namespace

template<primitive::BinaryOp op>
class MathBinaryBroadcastKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MathBinaryBroadcastKernel() = default;
  ~MathBinaryBroadcastKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);

    int64_t elem_cnt = tensor_z->shape().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream_ctx(), tensor_x->shape().NumAxes(), tensor_x->shape().ptr(),
                        tensor_x->dptr(), tensor_y->shape().NumAxes(), tensor_y->shape().ptr(),
                        tensor_y->dptr(), tensor_z->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define OP_BROADCAST_ELEMENTWISE_BINARY_PAIR_SEQ                                   \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_add", primitive::BinaryOp::kAdd)                 \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_sub", primitive::BinaryOp::kSub)                 \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_mul", primitive::BinaryOp::kMul)                 \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_div", primitive::BinaryOp::kDiv)                 \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_minimum", primitive::BinaryOp::kMin)             \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_maximum", primitive::BinaryOp::kMax)             \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_equal", primitive::BinaryOp::kEqual)             \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_not_equal", primitive::BinaryOp::kNotEqual)      \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater", primitive::BinaryOp::kLessThan)        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater_equal", primitive::BinaryOp::kLessEqual) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less", primitive::BinaryOp::kGreaterThan)        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less_equal", primitive::BinaryOp::kGreaterEqual) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_and", primitive::BinaryOp::kLogicalAnd)  \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_or", primitive::BinaryOp::kLogicalOr)    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_xor", primitive::BinaryOp::kLogicalXor)

#define REGISTER_MATH_BINARY_BROADCAST_KERNEL(op_name, binary_op) \
  REGISTER_USER_KERNEL(op_name)                                   \
      .SetCreateFn<MathBinaryBroadcastKernel<binary_op>>()        \
      .SetIsMatchedHob((BroadcastElementwiseBinaryPrimitiveExists<binary_op>() == true));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                     OP_BROADCAST_ELEMENTWISE_BINARY_PAIR_SEQ)

}  // namespace oneflow
