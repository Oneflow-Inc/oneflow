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
      const user_op::TensorDesc* targets = ctx->TensorDesc4ArgNameAndIndex("targets", 0);
      const user_op::TensorDesc* predictions = ctx->TensorDesc4ArgNameAndIndex("predictions", 0);
      CHECK_EQ_OR_RETURN(targets->shape().NumAxes(), 1);
      CHECK_GE_OR_RETURN(targets->data_type(), DataType::kInt32);
      CHECK_LE_OR_RETURN(targets->data_type(), DataType::kInt64);
      CHECK_EQ_OR_RETURN(predictions->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(predictions->data_type(), DataType::kFloat);
      const bool is_dynamic = targets->is_dynamic();
      CHECK_EQ_OR_RETURN(is_dynamic, predictions->is_dynamic());
      ctx->TensorDesc4ArgNameAndIndex("out", 0)->set_is_dynamic(is_dynamic);
      *ctx->Shape4ArgNameAndIndex("out", 0) = targets->shape();
      *ctx->Dtype4ArgNameAndIndex("out", 0) = kInt8;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("targets", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("predictions", 0), 0).Build();
      return Maybe<void>::Ok();
    });
}
