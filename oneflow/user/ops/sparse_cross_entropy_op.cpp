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

namespace {

Maybe<void> CheckPredictionLabelDesc(const user_op::TensorDesc* prediction_desc,
                                     const user_op::TensorDesc* label_desc) {
  CHECK_EQ_OR_RETURN(prediction_desc->is_dynamic(), label_desc->is_dynamic())
      << Error::RuntimeError()
      << "prediction and label are expected to have the same dynamic property, but found "
      << prediction_desc->is_dynamic() << " and " << label_desc->is_dynamic();
  CHECK_GE_OR_RETURN(prediction_desc->shape().NumAxes(), 2)
      << Error::RuntimeError()
      << "The dimension of prediction must be greater than or equal to 2, but found "
      << prediction_desc->shape().NumAxes();
  const int64_t num_out_axes = prediction_desc->shape().NumAxes() - 1;
  CHECK_EQ_OR_RETURN(label_desc->shape().NumAxes(), num_out_axes)
      << Error::RuntimeError()
      << "The dimension of label is expected to be less than that of prediction by 1, but found "
      << label_desc->shape().NumAxes() << " and " << num_out_axes;
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(prediction_desc->shape().At(i), label_desc->shape().At(i))
        << Error::RuntimeError() << "The size of prediction (" << prediction_desc->shape().At(i)
        << ") must match the size of label (" << label_desc->shape().At(i) << ") at dimension "
        << i;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  JUST(CheckPredictionLabelDesc(&prediction_desc, &label_desc));
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(prediction_desc.is_dynamic());
  out_desc->set_shape(label_desc.shape());
  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  JUST(CheckPredictionLabelDesc(&prediction_desc, &label_desc));
  CHECK_EQ_OR_RETURN(dy_desc.shape(), label_desc.shape())
      << Error::RuntimeError() << "The size of dy " << dy_desc.shape()
      << " must match the size of label " << label_desc.shape();
  ctx->SetOutputShape("prediction_diff", 0, prediction_desc.shape());
  ctx->SetOutputIsDynamic("prediction_diff", 0, prediction_desc.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_OR_RETURN(IsIndexDataType(label_desc.data_type()))
      << Error::TypeError() << "The dtype of label must be integer, but found "
      << DataType_Name(label_desc.data_type());
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(prediction_desc.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> InferDataTypeGrad(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_OR_RETURN(IsIndexDataType(label_desc.data_type()))
      << Error::TypeError() << "The dtype of label must be integer, but found "
      << DataType_Name(label_desc.data_type());
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), prediction_desc.data_type())
      << Error::TypeError() << "dy and prediction are expected to have the same dtype, but found "
      << DataType_Name(dy_desc.data_type()) << " and "
      << DataType_Name(prediction_desc.data_type());
  ctx->SetOutputDType("prediction_diff", 0, prediction_desc.data_type());
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> SparseCrossEntropyOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SparseCrossEntropyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDescFn(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyOp::InferDataType(user_op::InferContext* ctx) {
  return oneflow::InferDataType(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);
  CHECK_OR_RETURN(label_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  label_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SparseCrossEntropyMsOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& prediction =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), prediction.shape().NumAxes() - 1)
      .Broadcast(user_op::OpArg("label", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SparseCrossEntropyMsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDescFn(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyMsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyMsOp::InferDataType(user_op::InferContext* ctx) {
  return oneflow::InferDataType(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyMsOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);
  CHECK_OR_RETURN(label_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  label_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SparseCrossEntropyGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("prediction_diff", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SparseCrossEntropyGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferGradTensorDescFn(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataTypeGrad(ctx);
}

/*static*/ Maybe<void> SparseCrossEntropyMsGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& prediction =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("prediction_diff", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), prediction.shape().NumAxes() - 1)
      .Broadcast(user_op::OpArg("label", 0))
      .Broadcast(user_op::OpArg("dy", 0))
      .Split(user_op::OpArg("prediction_diff", 0), prediction.shape().NumAxes() - 1)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SparseCrossEntropyMsGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferGradTensorDescFn(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyMsGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SparseCrossEntropyMsGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataTypeGrad(ctx);
}

}  // namespace oneflow
