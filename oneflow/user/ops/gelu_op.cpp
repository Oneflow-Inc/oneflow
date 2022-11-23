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

Maybe<void> InferGeluTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferGeluDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetGeluSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ auto GeluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluTensorDesc(ctx);
}
/*static*/ auto GeluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluTensorDesc(ctx);
}
/*static*/ auto GeluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluDataType(ctx);
}
/*static*/ auto GeluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> { return GetGeluSbp(ctx); }

/*static*/ auto FastGeluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluTensorDesc(ctx);
}
/*static*/ auto FastGeluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluTensorDesc(ctx);
}
/*static*/ auto FastGeluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluDataType(ctx);
}
/*static*/ auto FastGeluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return GetGeluSbp(ctx);
}

namespace {

Maybe<void> InferGeluGradTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(dy_shape == x_shape)
      << "InferTensorDesc failed (" << ctx->op_name() << "). Expected x shape "
      << x_shape.ToString() << " to be equal to dy shape " << dy_shape.ToString();
  ctx->SetOutputShape("dx", 0, dy_shape);
  return Maybe<void>::Ok();
}

Maybe<void> InferGeluGradDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), ctx->InputDType("dy", 0))
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("dy", 0))
      << ", but got " << DataType_Name(ctx->InputDType("x", 0));
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetGeluGradSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ auto GeluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradTensorDesc(ctx);
}
/*static*/ auto GeluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradTensorDesc(ctx);
}
/*static*/ auto GeluGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradDataType(ctx);
}
/*static*/ auto GeluGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return GetGeluGradSbp(ctx);
}

/*static*/ auto FastGeluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradTensorDesc(ctx);
}
/*static*/ auto FastGeluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradTensorDesc(ctx);
}
/*static*/ auto FastGeluGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  return InferGeluGradDataType(ctx);
}
/*static*/ auto FastGeluGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return GetGeluGradSbp(ctx);
}

/*static*/ Maybe<void> FusedFastGeluMulOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& m_shape = ctx->InputShape("multiplier", 0);
  CHECK_OR_RETURN(ctx->InputShape("multiplier", 0) == in_shape)
      << "Expected multiplier shape " << in_shape.ToString() << ", but got " << m_shape.ToString();
  ctx->SetOutputShape("out", 0, in_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedFastGeluMulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> FusedFastGeluMulOp::InferDataType(user_op::InferContext* ctx) {
  const DataType in_dtype = ctx->InputDType("in", 0);
  const DataType m_dtype = ctx->InputDType("multiplier", 0);
  CHECK_EQ_OR_RETURN(m_dtype, in_dtype)
      << "Expected multiplier data type " << DataType_Name(in_dtype) << ", but got "
      << DataType_Name(m_dtype);
  ctx->SetOutputDType("out", 0, in_dtype);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedFastGeluMulOp::GetSbp(user_op::SbpContext* ctx) {
  return GetGeluSbp(ctx);
}

/*static*/ Maybe<void> FusedFastGeluMulGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& out_diff_shape = ctx->InputShape("out_diff", 0);
  const Shape& m_shape = ctx->InputShape("multiplier", 0);
  CHECK_EQ_OR_RETURN(out_diff_shape, in_shape);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(m_shape, in_shape);         // NOLINT(maybe-need-error-msg)
  ctx->SetOutputShape("in_diff", 0, in_shape);
  ctx->SetOutputShape("multiplier_diff", 0, m_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedFastGeluMulGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> FusedFastGeluMulGradOp::InferDataType(user_op::InferContext* ctx) {
  const DataType in_dtype = ctx->InputDType("in", 0);
  const DataType out_diff_dtype = ctx->InputDType("out_diff", 0);
  const DataType m_dtype = ctx->InputDType("multiplier", 0);
  CHECK_EQ_OR_RETURN(out_diff_dtype, in_dtype);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(m_dtype, in_dtype);         // NOLINT(maybe-need-error-msg)
  ctx->SetOutputDType("in_diff", 0, in_dtype);
  ctx->SetOutputDType("multiplier_diff", 0, m_dtype);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedFastGeluMulGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("in", 0))
      .Broadcast(user_op::OpArg("multiplier", 0))
      .PartialSum(user_op::OpArg("out_diff", 0))
      .PartialSum(user_op::OpArg("in_diff", 0))
      .PartialSum(user_op::OpArg("multiplier_diff", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
