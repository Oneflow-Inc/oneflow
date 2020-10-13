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

REGISTER_USER_OP("square_sum")
    .Input("x")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);

      *y->mut_shape() = Shape({1});
      *y->mut_data_type() = x->data_type();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("y", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t num_x_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, num_x_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("x", 0), i)
            .PartialSum(user_op::OpArg("y", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
