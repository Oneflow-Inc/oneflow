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
  const user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  const int64_t ndims = y->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), y->data_type(), ndims);
}

template<ep::primitive::BinaryOp op>
auto BroadcastElementwiseBinaryPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseBinaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastElementwiseBinaryPrimitive(&ctx, op).operator bool();
                          });
}

template<ep::primitive::BinaryOp op>
class ScalarByTensorKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ScalarByTensorKernel() = default;
  ~ScalarByTensorKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t elem_cnt = y->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), x->shape_view().NumAxes(), x->shape_view().ptr(), x->dptr(),
                        scalar->shape_view().NumAxes(), scalar->shape_view().ptr(), scalar->dptr(),
                        y->mut_dptr());
    } else {
      // For 0-size Tensor
      return;
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_SCALAR_BY_TENSOR_KERNEL(op_name, binary_op)                         \
  REGISTER_USER_KERNEL(op_name)                                                      \
      .SetCreateFn<ScalarByTensorKernel<binary_op>>()                                \
      .SetIsMatchedHob(BroadcastElementwiseBinaryPrimitiveExists<binary_op>())       \
      .SetInplaceProposalFn(                                                         \
          [](const user_op::InferContext&,                                           \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));           \
            return Maybe<void>::Ok();                                                \
          });

#define SCALAR_BY_TENSOR_SEQ                                                  \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add_by_tensor", ep::primitive::BinaryOp::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_sub_by_tensor", ep::primitive::BinaryOp::kSub) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul_by_tensor", ep::primitive::BinaryOp::kMul) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_div_by_tensor", ep::primitive::BinaryOp::kDiv)

OF_PP_FOR_EACH_TUPLE(REGISTER_SCALAR_BY_TENSOR_KERNEL, SCALAR_BY_TENSOR_SEQ)
}  // namespace oneflow
