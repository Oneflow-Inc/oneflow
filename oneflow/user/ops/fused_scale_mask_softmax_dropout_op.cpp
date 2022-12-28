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

/*static*/ auto FusedScaleMaskSoftmaxDropoutOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& mask_desc = ctx->InputTensorDesc("mask", 0);
  const auto x_shape = x_desc.shape();
  const auto mask_shape = mask_desc.shape();
  CHECK_EQ_OR_RETURN(x_desc.shape().At(x_shape.NumAxes() - 1),
                     mask_desc.shape().At(mask_shape.NumAxes() - 1))
      << " last dim of x and mask is not equal.";
  ctx->SetOutputShape("y", 0, x_desc.shape());
  ctx->SetOutputIsDynamic("y", 0, x_desc.is_dynamic());
  ctx->SetOutputShape("softmax_y", 0, x_desc.shape());
  ctx->SetOutputIsDynamic("softmax_y", 0, x_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return FusedScaleMaskSoftmaxDropoutOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& mask_desc = ctx->InputTensorDesc("mask", 0);
  CHECK_EQ_OR_RETURN(mask_desc.data_type(), DataType::kBool)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kBool) << ", but got "
      << DataType_Name(mask_desc.data_type());
  ctx->SetOutputDType("y", 0, x_desc.data_type());
  ctx->SetOutputDType("softmax_y", 0, x_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutOp::ModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&)
    -> Maybe<void> {
  user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
  user_op::InputArgModifier* dropout_mask_modifier = GetInputArgModifierFn("dropout_mask", 0);
  CHECK_OR_RETURN(mask_modifier != nullptr) << " cannot find mask input.";
  CHECK_OR_RETURN(dropout_mask_modifier != nullptr) << " cannot find dropout mask input.";
  mask_modifier->set_requires_grad(false);
  dropout_mask_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  CHECK_GE_OR_RETURN(x_tensor.shape().NumAxes(), 2) << " x num axes at least 2.";
  const user_op::TensorDesc& mask_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0);
  CHECK_EQ_OR_RETURN(x_tensor.shape().NumAxes(), mask_tensor.shape().NumAxes())
      << " x num axes must equal with mask.";
  FOR_RANGE(int64_t, axis, 0, x_tensor.shape().NumAxes() - 2) {
    // NOTE(chengcheng): mask support broadcast, when dim value = 1, sbp = broadcast
    if (mask_tensor.shape().At(axis) == 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis)
          .Broadcast(user_op::OpArg("mask", 0))
          .Split(user_op::OpArg("dropout_mask", 0), axis)
          .Split(user_op::OpArg("y", 0), axis)
          .Split(user_op::OpArg("softmax_y", 0), axis)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis)
          .Split(user_op::OpArg("mask", 0), axis)
          .Split(user_op::OpArg("dropout_mask", 0), axis)
          .Split(user_op::OpArg("y", 0), axis)
          .Split(user_op::OpArg("softmax_y", 0), axis)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskSoftmaxDropoutGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& softmax_y_desc = ctx->InputTensorDesc("softmax_y", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& mask_desc = ctx->InputTensorDesc("mask", 0);
  CHECK_EQ_OR_RETURN(dy_desc.shape(), softmax_y_desc.shape()) << " dy and y shape must equal.";
  CHECK_EQ_OR_RETURN(dy_desc.shape().At(dy_desc.shape().NumAxes() - 1),
                     mask_desc.shape().At(mask_desc.shape().NumAxes() - 1))
      << " last dim of y and mask is not equal.";
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_shape(dy_desc.shape());
  dx_desc->set_is_dynamic(dy_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedScaleMaskSoftmaxDropoutGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& softmax_y_desc = ctx->InputTensorDesc("softmax_y", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& mask_desc = ctx->InputTensorDesc("mask", 0);
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), softmax_y_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(softmax_y_desc.data_type())
      << ", but got " << DataType_Name(dy_desc.data_type());
  CHECK_EQ_OR_RETURN(mask_desc.data_type(), DataType::kBool)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kBool) << ", but got "
      << DataType_Name(mask_desc.data_type());
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_data_type(dy_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedScaleMaskSoftmaxDropoutGradOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  CHECK_GE_OR_RETURN(dy_tensor.shape().NumAxes(), 2) << " dy num axes at least 2.";
  const user_op::TensorDesc& mask_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0);
  CHECK_EQ_OR_RETURN(dy_tensor.shape().NumAxes(), mask_tensor.shape().NumAxes())
      << " dy num axes must equal with mask.";
  FOR_RANGE(int64_t, axis, 0, dy_tensor.shape().NumAxes() - 2) {
    if (mask_tensor.shape().At(axis) == 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("softmax_y", 0), axis)
          .Split(user_op::OpArg("dy", 0), axis)
          .Broadcast(user_op::OpArg("mask", 0))
          .Split(user_op::OpArg("dropout_mask", 0), axis)
          .Split(user_op::OpArg("dx", 0), axis)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("softmax_y", 0), axis)
          .Split(user_op::OpArg("dy", 0), axis)
          .Split(user_op::OpArg("mask", 0), axis)
          .Split(user_op::OpArg("dropout_mask", 0), axis)
          .Split(user_op::OpArg("dx", 0), axis)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
