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

REGISTER_USER_OP("in_top_k")
    .Input("targets")
    .Input("predictions")
    .Attr<int32_t>("k")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {

      const Shape* targets_shape = ctx->Shape4ArgNameAndIndex("targets", 0);
      CHECK_EQ_OR_RETURN(targets_shape->NumAxes(), 1);

      const Shape* predictions_shape = ctx->Shape4ArgNameAndIndex("predictions", 0);
      CHECK_EQ_OR_RETURN(predictions_shape->NumAxes(), 2);

      *ctx->Shape4ArgNameAndIndex("out", 0) = *targets_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = kInt8;

      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("targets", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("predictions", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });
}
