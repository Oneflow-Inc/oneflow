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

/*static*/ Maybe<void> SigmoidCrossEntropyOp::GetSbp(user_op::SbpContext* ctx) {
  const auto num_out_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("prediction", 0), i)
        .Split(user_op::OpArg("label", 0), i)
        .Split(user_op::OpArg("loss", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_EQ_OR_RETURN(label_desc.shape(), prediction_desc.shape())
      << Error::RuntimeError() << "The size of label " << label_desc.shape()
      << " must match the size of prediction " << prediction_desc.shape();
  user_op::TensorDesc* loss_desc = ctx->MutOutputTensorDesc("loss", 0);
  loss_desc->set_shape(prediction_desc.shape());
  loss_desc->set_is_dynamic(prediction_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SigmoidCrossEntropyOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("loss", 0, ctx->InputDType("prediction", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
  cond_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SigmoidCrossEntropyGradOp::GetSbp(user_op::SbpContext* ctx) {
  const auto num_dy_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("loss_diff", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_dy_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("loss_diff", 0), i)
        .Split(user_op::OpArg("label", 0), i)
        .Split(user_op::OpArg("prediction", 0), i)
        .Split(user_op::OpArg("prediction_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  const user_op::TensorDesc& loss_diff_desc = ctx->InputTensorDesc("loss_diff", 0);
  CHECK_EQ_OR_RETURN(label_desc.shape(), prediction_desc.shape())
      << Error::RuntimeError() << "The size of label " << label_desc.shape()
      << " must match the size of prediction " << prediction_desc.shape();
  CHECK_EQ_OR_RETURN(loss_diff_desc.shape(), prediction_desc.shape())
      << Error::RuntimeError() << "The size of loss_diff " << loss_diff_desc.shape()
      << " must match the size of prediction " << prediction_desc.shape();
  user_op::TensorDesc* prediction_diff = ctx->MutOutputTensorDesc("prediction_diff", 0);
  prediction_diff->set_shape(prediction_desc.shape());
  prediction_diff->set_is_dynamic(prediction_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SigmoidCrossEntropyGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("prediction_diff", 0, ctx->InputDType("prediction", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SigmoidCrossEntropyGradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
  cond_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
