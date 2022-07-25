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

/*static*/ Maybe<void> SmoothL1LossOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  FOR_RANGE(int64_t, i, 0, input_shape.NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SmoothL1LossOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic())
      << Error::RuntimeError()
      << "input and target are expected to have the same dynamic property, but found "
      << input_desc.is_dynamic() << " and " << target_desc.is_dynamic();
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << Error::RuntimeError() << "The size of input " << input_desc.shape()
      << " must match the size of target " << target_desc.shape();
  CHECK_GE_OR_RETURN(ctx->Attr<float>("beta"), 0)
      << Error::RuntimeError() << "beta must be greater than or equal to 0, but found it to be "
      << ctx->Attr<float>("beta");

  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *out_desc->mut_shape() = input_desc.shape();

  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SmoothL1LossOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SmoothL1LossOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << Error::TypeError() << "input and target are expected to have the same dtype, but found "
      << DataType_Name(input_desc.data_type()) << " and " << DataType_Name(target_desc.data_type());

  *ctx->MutOutputDType("out", 0) = ctx->InputDType("input", 0);

  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SmoothL1LossOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
  CHECK_OR_RETURN(target_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  target_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SmoothL1LossGradOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  FOR_RANGE(int64_t, i, 0, input_shape.NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("target", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SmoothL1LossGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  const auto& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic())
      << Error::RuntimeError()
      << "input and target are expected to have the same dynamic property, but found "
      << input_desc.is_dynamic() << " and " << target_desc.is_dynamic();
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << Error::RuntimeError() << "The size of input " << input_desc.shape()
      << " must match the size of target " << target_desc.shape();
  CHECK_EQ_OR_RETURN(dy_desc.shape(), target_desc.shape())
      << Error::RuntimeError() << "The size of dy " << dy_desc.shape()
      << " must match the size of target " << target_desc.shape();

  CHECK_GE_OR_RETURN(ctx->Attr<float>("beta"), 0)
      << Error::RuntimeError() << "beta must be greater than or equal to 0, but found it to be "
      << ctx->Attr<float>("beta");

  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *dx_desc->mut_shape() = input_desc.shape();

  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SmoothL1LossGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SmoothL1LossGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << Error::TypeError() << "input and target are expected to have the same dtype, but found "
      << DataType_Name(input_desc.data_type()) << " and " << DataType_Name(target_desc.data_type());

  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("dy", 0);

  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("smooth_l1_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("smooth_l1_loss_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Input("input", op.input("input", 0))
                .Input("target", op.input("target", 0))
                .Output("dx")
                .Attr("beta", op.attr<float>("beta"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "input", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
