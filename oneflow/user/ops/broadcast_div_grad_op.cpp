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

/* static */ Maybe<void> BroadcastDivGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dy", 0, ctx->InputShape("y", 0));
  ctx->SetOutputIsDynamic("dy", 0, ctx->InputIsDynamic("y", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadcastDivGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BroadcastDivGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const Shape& z_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape();
  CHECK_LE_OR_RETURN(y_shape.NumAxes(), z_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
    const int64_t axis_y = y_shape.NumAxes() - 1 - i;
    const int64_t axis_z = z_shape.NumAxes() - 1 - i;
    if (y_shape.At(axis_y) == z_shape.At(axis_z)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("y", 0), axis_y)
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .Split(user_op::OpArg("dy", 0), axis_y)
          .Build();
    } else {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("y", 0))
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .PartialSum(user_op::OpArg("dy", 0))
          .Build();
    }
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Broadcast(user_op::OpArg("dz", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("y", 0))
      .Broadcast(user_op::OpArg("z", 0))
      .PartialSum(user_op::OpArg("dz", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastDivGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dy", 0, ctx->InputDType("y", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
