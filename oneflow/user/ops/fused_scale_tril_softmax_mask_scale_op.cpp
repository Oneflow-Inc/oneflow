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
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  ctx->SetOutputShape("y", 0, x_desc.shape());
  ctx->SetOutputIsDynamic("y", 0, x_desc.is_dynamic());
  ctx->SetOutputShape("softmax_y", 0, x_desc.shape());
  ctx->SetOutputIsDynamic("softmax_y", 0, x_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedTrilScaleSoftmaxMaskScaleOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  ctx->SetOutputDType("y", 0, x_desc.data_type());
  ctx->SetOutputDType("softmax_y", 0, x_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleOp::ModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&)
    -> Maybe<void> {
  user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
  CHECK_OR_RETURN(mask_modifier != nullptr);
  mask_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  CHECK_GE_OR_RETURN(x_tensor.shape().NumAxes(), 2);
  FOR_RANGE(int64_t, axis, 0, x_tensor.shape().NumAxes() - 2) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), axis)
        .Split(user_op::OpArg("mask", 0), axis)
        .Split(user_op::OpArg("y", 0), axis)
        .Split(user_op::OpArg("softmax_y", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedTrilScaleSoftmaxMaskScaleGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& softmax_y_desc = ctx->InputTensorDesc("softmax_y", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  CHECK_OR_RETURN(dy_desc.shape() == softmax_y_desc.shape());
  dx_desc->set_shape(dy_desc.shape());
  dx_desc->set_is_dynamic(dy_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedTrilScaleSoftmaxMaskScaleGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& softmax_y_desc = ctx->InputTensorDesc("softmax_y", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  CHECK_OR_RETURN(dy_desc.data_type() == softmax_y_desc.data_type());
  dx_desc->set_data_type(dy_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ auto FusedTrilScaleSoftmaxMaskScaleGradOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  CHECK_GE_OR_RETURN(dy_tensor.shape().NumAxes(), 2);
  FOR_RANGE(int64_t, axis, 0, dy_tensor.shape().NumAxes() - 2) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("softmax_y", 0), axis)
        .Split(user_op::OpArg("dy", 0), axis)
        .Split(user_op::OpArg("mask", 0), axis)
        .Split(user_op::OpArg("dx", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
