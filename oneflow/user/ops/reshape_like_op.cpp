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
#include "oneflow/user/ops/reshape_user_op_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> ReshapeLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const auto& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("like", 0))
      .Broadcast(user_op::OpArg("in", 0))
      .Broadcast(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  user_op::UserOpSbpSignatureBuilder builder = ctx->NewBuilder();
  return ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(
      in_shape, like_shape, {{"in", 0}}, {{"like", 0}, {"out", 0}}, ctx->parallel_num(), &builder);
}
/*static*/ Maybe<void> ReshapeLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& like_shape = ctx->InputShape("like", 0);
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), like_shape.elem_cnt())
      << Error::RuntimeError()
      << "The element number of the in tensor must be equal to the element number of the "
         "like tensor, "
      << "but got " << in_shape.elem_cnt() << " and " << like_shape.elem_cnt();
  *ctx->MutOutputShape("out", 0) = like_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReshapeLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReshapeLikeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReshapeLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_NOTNULL_OR_RETURN(like_modifier);  // NOLINT(maybe-need-error-msg)
  like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("reshape_like")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        const auto& in_desc = op.TensorDesc4ArgNameAndIndex("in", 0);
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        if (in_desc.is_dynamic()) {
          user_op::UserOpConfWrapper reshape_grad_op =
              builder.Op("reshape_like")
                  .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                  .Input("like", op.input("in", 0))
                  .Output("out")
                  .Build();
          op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
          AddOp(reshape_grad_op);
        } else {
          user_op::UserOpConfWrapper reshape_grad_op =
              builder.Op("reshape")
                  .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                  .Attr("shape", in_desc.shape())
                  .Output("out")
                  .Build();
          op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
          AddOp(reshape_grad_op);
        }
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
