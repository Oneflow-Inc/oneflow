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
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_EQ_OR_RETURN(targets->shape().NumAxes(), 1);
      CHECK_OR_RETURN(IsIndexDataType(targets->data_type()));
      CHECK_EQ_OR_RETURN(predictions->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(predictions->data_type(), DataType::kFloat);
      const bool is_dynamic = targets->is_dynamic();
      CHECK_EQ_OR_RETURN(is_dynamic, predictions->is_dynamic());
      out->set_is_dynamic(is_dynamic);
      *out->mut_shape() = targets->shape();
      *out->mut_data_type() = kInt8;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });
}
