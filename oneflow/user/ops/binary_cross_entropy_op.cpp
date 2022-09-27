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

Maybe<void> InferTensorDescFn_(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.is_dynamic(), input_desc.is_dynamic());
    CHECK_EQ_OR_RETURN(weight_desc.shape(), input_desc.shape());
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(input_desc.is_dynamic());
  out_desc->set_shape(input_desc.shape());

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType_(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(input_desc.data_type()) << ", but got "
      << DataType_Name(target_desc.data_type());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.data_type(), input_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(input_desc.data_type())
        << ", but got " << DataType_Name(weight_desc.data_type());
  }

  ctx->SetOutputDType("out", 0, ctx->InputDType("input", 0));

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

  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_is_dynamic(input_desc.is_dynamic());
  dx_desc->set_shape(input_desc.shape());

  return Maybe<void>::Ok();
}
Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(input_desc.data_type()) << ", but got "
      << DataType_Name(target_desc.data_type());
  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.data_type(), input_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(input_desc.data_type())
        << ", but got " << DataType_Name(weight_desc.data_type());
  }

  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));

  return Maybe<void>::Ok();
}
}  // namespace

/* static */ Maybe<void> BinaryCrossEntropyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDescFn_(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyOp::GetSbp(user_op::SbpContext* ctx) {
  return GenLossForwardDefaultGetSbpFn()(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
  CHECK_OR_RETURN(target_modifier != nullptr);
  target_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BinaryCrossEntropyOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType_(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferGradTensorDescFn(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyGradOp::GetSbp(user_op::SbpContext* ctx) {
  return GenLossBackwardDefaultGetSbpFn()(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferGradDataType(ctx);
}

}  // namespace oneflow
