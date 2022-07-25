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
#include "oneflow/user/ops/loss_op_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {
namespace {
Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.is_dynamic(), input_desc.is_dynamic());
    CHECK_EQ_OR_RETURN(weight_desc.shape(), input_desc.shape());
  }
  if (ctx->Attr<bool>("has_pos_weight")) {
    const auto& pos_weight_desc = ctx->InputTensorDesc("pos_weight", 0);
    CHECK_EQ_OR_RETURN(pos_weight_desc.is_dynamic(), input_desc.is_dynamic());
    CHECK_EQ_OR_RETURN(pos_weight_desc.shape(),
                       Shape({input_desc.shape().At(input_desc.shape().NumAxes() - 1)}));
  }
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *out_desc->mut_shape() = input_desc.shape();

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType_(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.data_type(), input_desc.data_type());
  }
  if (ctx->Attr<bool>("has_pos_weight")) {
    const auto& pos_weight_desc = ctx->InputTensorDesc("pos_weight", 0);
    CHECK_EQ_OR_RETURN(pos_weight_desc.data_type(), input_desc.data_type());
  }
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("input", 0);

  return Maybe<void>::Ok();
}
Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  const auto& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape());
  CHECK_EQ_OR_RETURN(dy_desc.shape(), target_desc.shape());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.is_dynamic(), input_desc.is_dynamic());
    CHECK_EQ_OR_RETURN(weight_desc.shape(), input_desc.shape());
  }
  if (ctx->Attr<bool>("has_pos_weight")) {
    const auto& pos_weight_desc = ctx->InputTensorDesc("pos_weight", 0);
    CHECK_EQ_OR_RETURN(pos_weight_desc.is_dynamic(), input_desc.is_dynamic());
    CHECK_EQ_OR_RETURN(pos_weight_desc.shape(),
                       Shape({input_desc.shape().At(input_desc.shape().NumAxes() - 1)}));
  }

  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *dx_desc->mut_shape() = input_desc.shape();

  return Maybe<void>::Ok();
}
Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.data_type(), input_desc.data_type());
  }
  if (ctx->Attr<bool>("has_pos_weight")) {
    const auto& pos_weight_desc = ctx->InputTensorDesc("pos_weight", 0);
    CHECK_EQ_OR_RETURN(pos_weight_desc.data_type(), input_desc.data_type());
  }
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("dy", 0);

  return Maybe<void>::Ok();
}
}  // namespace

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDescFn(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyWithLogitsOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsOp::GetSbp(user_op::SbpContext* ctx) {
  return GenLossForwardDefaultGetSbpFn(
      [](user_op::UserOpSbpSignatureBuilder& builder, user_op::SbpContext* ctx) {
        if (ctx->user_op_conf().has_input("pos_weight", 0)) {
          builder.Broadcast(user_op::OpArg("pos_weight", 0));
        }
      })(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
  CHECK_OR_RETURN(target_modifier != nullptr);
  target_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType_(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferGradTensorDescFn(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyWithLogitsGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsGradOp::GetSbp(user_op::SbpContext* ctx) {
  return GenLossBackwardDefaultGetSbpFn(
      [](user_op::UserOpSbpSignatureBuilder& builder, user_op::SbpContext* ctx) {
        if (ctx->user_op_conf().has_input("pos_weight", 0)) {
          builder.Broadcast(user_op::OpArg("pos_weight", 0));
        }
      })(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsGradOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferGradDataType(ctx);
}

REGISTER_USER_OP_GRAD("binary_cross_entropy_with_logits")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        builder.Op("binary_cross_entropy_with_logits_grad")
            .Input("input", op.input("input", 0))
            .Input("target", op.input("target", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx");
        if (op.user_op_conf().has_input("weight", 0)) {
          builder.Input("weight", op.input("weight", 0));
        }
        if (op.attr<bool>("has_pos_weight")) {
          builder.Input("pos_weight", op.input("pos_weight", 0))
              .Attr("has_pos_weight", op.attr<bool>("has_pos_weight"));
        }
        user_op::UserOpConfWrapper grad_op = builder.Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "input", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
