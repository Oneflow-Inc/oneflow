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
    .Input("predictions")
    .Input("targets")
    .Attr("k", UserOpAttrType::kAtInt32)
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {

      const Shape* pre_shape = ctx->Shape4ArgNameAndIndex("predictions", 0);
      CHECK_EQ_OR_RETURN(pre_shape->NumAxes(), 2);

      const Shape* target_shape = ctx->Shape4ArgNameAndIndex("targets", 0);
      CHECK_EQ_OR_RETURN(target_shape->NumAxes(), 1);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *target_shape;

      *ctx->Dtype4ArgNameAndIndex("out", 0) = kInt8;

      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("targets", 0);
      return Maybe<void>::Ok();
    });
}