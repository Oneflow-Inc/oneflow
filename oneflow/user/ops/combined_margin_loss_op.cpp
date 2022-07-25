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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> CombinedMarginLossOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& label = ctx->InputTensorDesc("label", 0);
  user_op::TensorDesc* theta = ctx->OutputTensorDesc("theta", 0);
  CHECK_EQ_OR_RETURN(label.shape().At(0), x.shape().At(0));
  CHECK_GE_OR_RETURN(x.shape().NumAxes(), 2);
  *ctx->MutOutputShape("y", 0) = ctx->InputShape("x", 0);
  *ctx->MutIsDynamic4ArgNameAndIndex("y", 0) = ctx->InputIsDynamic("x", 0);
  *theta->mut_is_dynamic() = x.is_dynamic();
  *theta->mut_shape() = label.shape();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CombinedMarginLossOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CombinedMarginLossOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("theta", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 1)
      .Broadcast(user_op::OpArg("label", 0))
      .Split(user_op::OpArg("y", 0), 1)
      .PartialSum(user_op::OpArg("theta", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CombinedMarginLossOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* label_arg_modifier = GetInputArgModifierFn("label", 0);
  label_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CombinedMarginLossOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("y", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("theta", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CombinedMarginLossGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& label = ctx->InputTensorDesc("label", 0);
  const user_op::TensorDesc& theta = ctx->InputTensorDesc("theta", 0);
  CHECK_EQ_OR_RETURN(label.shape().At(0), dy.shape().At(0));
  CHECK_EQ_OR_RETURN(label.shape().At(0), theta.shape().At(0));
  CHECK_GE_OR_RETURN(dy.shape().NumAxes(), 2);
  *ctx->MutOutputShape("dx", 0) = ctx->InputShape("dy", 0);
  *ctx->MutIsDynamic4ArgNameAndIndex("dx", 0) = ctx->InputIsDynamic("dy", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CombinedMarginLossGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CombinedMarginLossGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("theta", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 1)
      .Broadcast(user_op::OpArg("label", 0))
      .Broadcast(user_op::OpArg("theta", 0))
      .Split(user_op::OpArg("dx", 0), 1)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CombinedMarginLossGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("combined_margin_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("combined_margin_loss_grad")
                                                 .Input("label", op.input("label", 0))
                                                 .Input("theta", op.output("theta", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("dx")
                                                 .Attr("m1", op.attr<float>("m1"))
                                                 .Attr("m2", op.attr<float>("m2"))
                                                 .Attr("m3", op.attr<float>("m3"))
                                                 .Attr("depth", op.attr<int64_t>("depth"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
