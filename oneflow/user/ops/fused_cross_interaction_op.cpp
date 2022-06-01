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

/* static */ Maybe<void> FusedCrossInteractionOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  CHECK_EQ_OR_RETURN(x_shape.At(1), weight_shape.At(1)) << "Matmul K dims should be equal. ";
  *ctx->OutputShape("matmul_result", 0) = Shape({x_shape.At(0), weight_shape.At(0)});
  const Shape& x0_shape = ctx->InputShape("x_0", 0);
  const Shape& bias_shape = ctx->InputShape("bias", 0);
  CHECK_EQ_OR_RETURN(bias_shape.At(0), x0_shape.At(1)) << "Bias dim should be equal to X0 dim1. ";
  *ctx->OutputShape("out", 0) = x0_shape;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossInteractionOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedCrossInteractionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("x_0", 0), 0)
      .Broadcast(user_op::OpArg("bias", 0))
      .Split(user_op::OpArg("matmul_result", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossInteractionOp::InferDataType(user_op::InferContext* ctx) {
  // TODO add check
  *ctx->OutputDType("out", 0) = ctx->InputDType("x", 0);
  *ctx->OutputDType("matmul_result", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossInteractionGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x0_shape = ctx->InputShape("x_0", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  *ctx->OutputShape("dx_0", 0) = x0_shape;
  *ctx->OutputShape("dw", 0) = weight_shape;
  *ctx->OutputShape("dx", 0) = x0_shape;
  *ctx->OutputShape("dbias", 0) = Shape({x0_shape.At(1)});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossInteractionGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedCrossInteractionGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("x_0", 0), 0)
      .Split(user_op::OpArg("matmul_result", 0), 0)
      .Split(user_op::OpArg("dx_0", 0), 0)
      .Broadcast(user_op::OpArg("dw", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .Broadcast(user_op::OpArg("dbias", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossInteractionGradOp::InferDataType(user_op::InferContext* ctx) {
  // TODO add check
  *ctx->OutputDType("dx_0", 0) = ctx->InputDType("x", 0);
  *ctx->OutputDType("dw", 0) = ctx->InputDType("x", 0);
  *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
  *ctx->OutputDType("dbias", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
