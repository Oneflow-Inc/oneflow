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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> BroadcastPowXGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("dx", 0) = ctx->InputShape("x", 0);
  *ctx->MutOutputIsDynamic("dx", 0) = ctx->InputIsDynamic("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadcastPowXGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BroadcastPowXGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const Shape& z_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape();
  CHECK_LE_OR_RETURN(x_shape.NumAxes(), z_shape.NumAxes());
  CHECK_LE_OR_RETURN(y_shape.NumAxes(), z_shape.NumAxes());
  int64_t min_num_axes = std::min(x_shape.NumAxes(), y_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, min_num_axes) {
    const int64_t axis_x = x_shape.NumAxes() - 1 - i;
    const int64_t axis_y = y_shape.NumAxes() - 1 - i;
    const int64_t axis_z = z_shape.NumAxes() - 1 - i;
    if (z_shape.At(axis_z) == x_shape.At(axis_x) && z_shape.At(axis_z) == y_shape.At(axis_y)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis_x)
          .Split(user_op::OpArg("y", 0), axis_y)
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .Split(user_op::OpArg("dx", 0), axis_x)
          .Build();
    } else if (z_shape.At(axis_z) == y_shape.At(axis_y)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("y", 0), axis_y)
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .PartialSum(user_op::OpArg("dx", 0))
          .Build();
    } else if (z_shape.At(axis_z) == x_shape.At(axis_x)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis_x)
          .Broadcast(user_op::OpArg("y", 0))
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .Split(user_op::OpArg("dx", 0), axis_x)
          .Build();
    }
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Broadcast(user_op::OpArg("dz", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("y", 0))
      .Broadcast(user_op::OpArg("z", 0))
      .Broadcast(user_op::OpArg("dz", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("y", 0))
      .Broadcast(user_op::OpArg("z", 0))
      .PartialSum(user_op::OpArg("dz", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastPowXGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastPowYGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("dy", 0) = ctx->InputShape("y", 0);
  *ctx->MutOutputIsDynamic("dy", 0) = ctx->InputIsDynamic("y", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadcastPowYGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BroadcastPowYGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const Shape& z_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape();
  CHECK_LE_OR_RETURN(x_shape.NumAxes(), z_shape.NumAxes());
  int64_t min_num_axes = std::min(x_shape.NumAxes(), y_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, min_num_axes) {
    const int64_t axis_x = x_shape.NumAxes() - 1 - i;
    const int64_t axis_y = y_shape.NumAxes() - 1 - i;
    const int64_t axis_z = z_shape.NumAxes() - 1 - i;
    if (z_shape.At(axis_z) == x_shape.At(axis_x) && z_shape.At(axis_z) == y_shape.At(axis_y)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis_x)
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("y", 0), axis_y)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .Split(user_op::OpArg("dy", 0), axis_y)
          .Build();
    } else if (z_shape.At(axis_z) == y_shape.At(axis_y)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("y", 0), axis_y)
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .Split(user_op::OpArg("dy", 0), axis_y)
          .Build();
    } else if (z_shape.At(axis_z) == x_shape.At(axis_x)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis_x)
          .Broadcast(user_op::OpArg("y", 0))
          .Split(user_op::OpArg("z", 0), axis_z)
          .Split(user_op::OpArg("dz", 0), axis_z)
          .PartialSum(user_op::OpArg("dy", 0))
          .Build();
    }
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Broadcast(user_op::OpArg("dz", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .Broadcast(user_op::OpArg("z", 0))
      .PartialSum(user_op::OpArg("dz", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastPowYGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dy", 0) = ctx->InputDType("y", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
