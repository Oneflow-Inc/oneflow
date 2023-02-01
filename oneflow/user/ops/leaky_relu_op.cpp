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

/* static */ Maybe<void> LeakyReluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("y", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LeakyReluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LeakyReluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LeakyReluOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LeakyReluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(dy_shape == x_shape);
  ctx->SetOutputShape("dx", 0, dy_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LeakyReluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LeakyReluGradOp::GetSbp(user_op::SbpContext* ctx) {
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

/* static */ Maybe<void> LeakyReluGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), ctx->InputDType("dy", 0))
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("dy", 0))
      << ", but got " << DataType_Name(ctx->InputDType("x", 0));
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
