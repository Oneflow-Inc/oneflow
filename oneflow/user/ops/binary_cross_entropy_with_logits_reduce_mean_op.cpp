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

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  *out_desc->mut_is_dynamic() = false;
  *out_desc->mut_shape() = Shape({});
  return Maybe<void>::Ok();
}

Maybe<void> InferFwDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << "Input datatype should be equal to Target datatype. ";
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("input", 0);

  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = false;
  *dx_desc->mut_shape() = input_desc.shape();
  return Maybe<void>::Ok();
}

Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << "Input datatype should be equal to Target datatype. ";
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("dy", 0);
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

REGISTER_USER_OP_GRAD("binary_cross_entropy_with_logits_reduce_mean")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        builder.Op("binary_cross_entropy_with_logits_reduce_mean_grad")
            .Input("input", op.input("input", 0))
            .Input("target", op.input("target", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx");
        user_op::UserOpConfWrapper grad_op = builder.Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "input", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

/* static */ Maybe<void> FusedBCEReduceMeanFwBwOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape())
      << "Input shape should be equal to Target shape. ";
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  *out_desc->mut_is_dynamic() = false;
  *out_desc->mut_shape() = Shape({});
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = false;
  *dx_desc->mut_shape() = input_desc.shape();
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
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type())
      << "Input datatype should be equal to Target datatype. ";
  DataType out_dtype = ctx->Attr<DataType>("out_dtype");
  if (out_dtype == DataType::kInvalidDataType) { out_dtype = input_desc.data_type(); }
  *ctx->MutOutputDType("out", 0) = out_dtype;
  *ctx->MutOutputDType("dx", 0) = input_desc.data_type();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
