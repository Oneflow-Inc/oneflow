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

/*static*/ Maybe<void> PreluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const user_op::TensorDesc& alpha_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
  if (alpha_tensor.shape().At(0) != 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 1)
        .Split(user_op::OpArg("alpha", 0), 0)
        .Split(user_op::OpArg("y", 0), 1)
        .Build();
  }
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    if (i == 1) continue;
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Broadcast(user_op::OpArg("alpha", 0))
        .Split(user_op::OpArg("y", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PreluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& alpha_shape = ctx->InputShape("alpha", 0);
  CHECK_EQ_OR_RETURN(alpha_shape.NumAxes(), 1);
  ctx->SetOutputShape("y", 0, x_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PreluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> PreluOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> PreluGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("alpha", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("alpha", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 1)
      .Split(user_op::OpArg("x", 0), 1)
      .Split(user_op::OpArg("alpha", 0), 0)
      .Split(user_op::OpArg("dx", 0), 1)
      .Split(user_op::OpArg("alpha_diff", 0), 0)
      .Build();
  FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("alpha", 0), 0)
        .Split(user_op::OpArg("dx", 0), i)
        .Split(user_op::OpArg("alpha_diff", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PreluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const Shape& alpha_shape = ctx->InputShape("alpha", 0);
  CHECK_EQ_OR_RETURN(alpha_shape.NumAxes(), 1);
  CHECK_OR_RETURN((alpha_shape.At(0) == x_shape.At(1)) || (alpha_shape.At(0) == 1));
  CHECK_EQ_OR_RETURN(dy_shape, x_shape);
  ctx->SetOutputShape("dx", 0, x_shape);
  ctx->SetOutputShape("alpha_diff", 0, alpha_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PreluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> PreluGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  ctx->SetOutputDType("alpha_diff", 0, ctx->InputDType("alpha", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
