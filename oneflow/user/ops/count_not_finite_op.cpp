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

REGISTER_USER_OP("count_not_finite")
    .Input("x")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y_desc->mut_shape() = Shape({1});
      *y_desc->mut_data_type() = DataType::kInt64;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("x", 0), i)
            .PartialSum(user_op::OpArg("y", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("multi_count_not_finite")
    .InputWithMinimum("x", 1)
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* first_x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      for (const auto& in_arg_pair : ctx->inputs()) {
        const user_op::TensorDesc* x_desc =
            ctx->TensorDesc4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
        CHECK_EQ_OR_RETURN(x_desc->data_type(), first_x_desc->data_type());
      }
      user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y_desc->mut_shape() = Shape({1});
      *y_desc->mut_data_type() = DataType::kInt64;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(user_op::OpArg("y", 0)).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
