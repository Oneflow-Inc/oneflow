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

namespace oneflow {
REGISTER_USER_OP("broadcast_pow_x_grad")
    .Input("x")
    .Input("y")
    .Input("z")
    .Input("dz")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("dx", 0) = ctx->InputShape("x", 0);
      *ctx->OutputIsDynamic("dx", 0) = ctx->InputIsDynamic("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
      const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
      const Shape& z_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape();
      CHECK_LE_OR_RETURN(x_shape.NumAxes(), z_shape.NumAxes());
      CHECK_LE_OR_RETURN(y_shape.NumAxes(), z_shape.NumAxes());
      FOR_RANGE(int64_t, i, 0, z_shape.NumAxes()) {
        const int64_t _axis = z_shape.NumAxes() - 1 - i;
        if (z_shape.At(_axis) == x_shape.At(_axis) && z_shape.At(_axis) == y_shape.At(_axis)) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("x", 0), _axis)
              .Split(user_op::OpArg("y", 0), _axis)
              .Split(user_op::OpArg("z", 0), _axis)
              .Split(user_op::OpArg("dz", 0), _axis)
              .Split(user_op::OpArg("dx", 0), _axis)
              .Build();
        }
      }
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("y", 0))
          .PartialSum(user_op::OpArg("z", 0))
          .Broadcast(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("dx", 0))
          .Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("y", 0))
          .Broadcast(user_op::OpArg("z", 0))
          .Broadcast(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("dx", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("y", 0))
          .Broadcast(user_op::OpArg("z", 0))
          .PartialSum(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("dx", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("broadcast_pow_y_grad")
    .Input("x")
    .Input("y")
    .Input("z")
    .Input("dz")
    .Output("dy")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("dy", 0) = ctx->InputShape("y", 0);
      *ctx->OutputIsDynamic("dy", 0) = ctx->InputIsDynamic("y", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
      const Shape& z_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape();
      CHECK_LE_OR_RETURN(x_shape.NumAxes(), z_shape.NumAxes());
      FOR_RANGE(int64_t, i, 0, z_shape.NumAxes()) {
        const int64_t _axis = z_shape.NumAxes() - 1 - i;
        if (z_shape.At(_axis) == x_shape.At(_axis)) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("x", 0), _axis)
              .Split(user_op::OpArg("z", 0), _axis)
              .Split(user_op::OpArg("dz", 0), _axis)
              .Split(user_op::OpArg("dy", 0), _axis)
              .Build();
        }
      }
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("z", 0))
          .Broadcast(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("dy", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("z", 0))
          .PartialSum(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("dy", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dy", 0) = ctx->InputDType("y", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
