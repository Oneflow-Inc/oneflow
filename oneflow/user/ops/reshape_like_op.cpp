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
  return ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(in_shape, like_shape, {{"in", 0}},
                                                          {{"like", 0}, {"out", 0}},
                                                          ctx->hierarchy_value(), &builder);
}
/*static*/ Maybe<void> ReshapeLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& like_shape = ctx->InputShape("like", 0);
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), like_shape.elem_cnt())
      << Error::RuntimeError()
      << "The element number of the in tensor must be equal to the element number of the "
         "like tensor, "
      << "but got " << in_shape.elem_cnt() << " and " << like_shape.elem_cnt();
  ctx->SetOutputShape("out", 0, like_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReshapeLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReshapeLikeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReshapeLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_NOTNULL_OR_RETURN(like_modifier);  // NOLINT(maybe-need-error-msg)
  like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
