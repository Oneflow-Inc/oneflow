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
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewBroadcastElementwiseBinaryPrimitive(
    Context* ctx, ep::primitive::BinaryOp op) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* z = ctx->TensorDesc4ArgNameAndIndex("z", 0);
  const int64_t ndims = z->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), z->data_type(), ndims);
}

template<ep::primitive::BinaryOp op>
auto BroadcastElementwiseBinaryPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseBinaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastElementwiseBinaryPrimitive(&ctx, op).operator bool();
                          });
}

}  // namespace

template<ep::primitive::BinaryOp op>
class MathBinaryBroadcastKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MathBinaryBroadcastKernel() = default;
  ~MathBinaryBroadcastKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    if (tensor_z->shape().elem_cnt() != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), tensor_x->shape().NumAxes(), tensor_x->shape().ptr(),
                        tensor_x->dptr(), tensor_y->shape().NumAxes(), tensor_y->shape().ptr(),
                        tensor_y->dptr(), tensor_z->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define OP_BROADCAST_ELEMENTWISE_BINARY_PAIR_SEQ                                          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_add", ep::primitive::BinaryOp::kAdd)                    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_sub", ep::primitive::BinaryOp::kSub)                    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_mul", ep::primitive::BinaryOp::kMul)                    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_div", ep::primitive::BinaryOp::kDiv)                    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_minimum", ep::primitive::BinaryOp::kMin)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_maximum", ep::primitive::BinaryOp::kMax)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_equal", ep::primitive::BinaryOp::kEqual)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_not_equal", ep::primitive::BinaryOp::kNotEqual)         \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater", ep::primitive::BinaryOp::kGreaterThan)        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater_equal", ep::primitive::BinaryOp::kGreaterEqual) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less", ep::primitive::BinaryOp::kLessThan)              \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less_equal", ep::primitive::BinaryOp::kLessEqual)       \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_and", ep::primitive::BinaryOp::kLogicalAnd)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_or", ep::primitive::BinaryOp::kLogicalOr)       \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_xor", ep::primitive::BinaryOp::kLogicalXor)

#define REGISTER_MATH_BINARY_BROADCAST_KERNEL(op_name, binary_op) \
  REGISTER_USER_KERNEL(op_name)                                   \
      .SetCreateFn<MathBinaryBroadcastKernel<binary_op>>()        \
      .SetIsMatchedHob((BroadcastElementwiseBinaryPrimitiveExists<binary_op>()));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                     OP_BROADCAST_ELEMENTWISE_BINARY_PAIR_SEQ)

}  // namespace oneflow
