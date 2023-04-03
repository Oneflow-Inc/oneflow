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

#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(false);
  out_desc->set_shape(Shape({}));
  return Maybe<void>::Ok();
}

Maybe<void> InferFwDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_GE_OR_RETURN(DType::priority_order[input_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  CHECK_GE_OR_RETURN(DType::priority_order[target_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  ctx->SetOutputDType("out", 0, ctx->InputDType("target", 0));

  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_is_dynamic(false);
  dx_desc->set_shape(input_desc.shape());
  return Maybe<void>::Ok();
}

Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_GE_OR_RETURN(DType::priority_order[input_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  CHECK_GE_OR_RETURN(DType::priority_order[target_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  ctx->SetOutputDType("dx", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}
}  // namespace

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDescFn(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanOp::GetSbp(
    user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("target", 0), 0)
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
  CHECK_OR_RETURN(target_modifier != nullptr) << "target_modifier should not be nullptr. ";
  target_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferFwDataType(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferGradTensorDescFn(ctx);
}

/*static*/ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradOp::GetSbp(
    user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("target", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Broadcast(user_op::OpArg("dy", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferGradDataType(ctx);
}

/* static */ Maybe<void> FusedBCEReduceMeanFwBwOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(false);
  out_desc->set_shape(Shape({}));
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_is_dynamic(false);
  dx_desc->set_shape(input_desc.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedBCEReduceMeanFwBwOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedBCEReduceMeanFwBwOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("target", 0), 0)
      .PartialSum(user_op::OpArg("out", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedBCEReduceMeanFwBwOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_GE_OR_RETURN(DType::priority_order[input_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  CHECK_GE_OR_RETURN(DType::priority_order[target_desc.data_type()],
                     DType::priority_order[DType::Float16()->data_type()]);
  DataType out_dtype = ctx->Attr<DataType>("out_dtype");
  if (out_dtype == DataType::kInvalidDataType) { out_dtype = target_desc.data_type(); }
  ctx->SetOutputDType("out", 0, out_dtype);
  ctx->SetOutputDType("dx", 0, input_desc.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
