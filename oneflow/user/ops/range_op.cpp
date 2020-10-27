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
REGISTER_USER_OP("range")
    .Output("out")
    .Attr<int64_t>("start")
    .Attr<int64_t>("delta")
    .Attr<int64_t>("range_shape")
    .Attr<DataType>("dtype")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      auto dtype = ctx->Attr<DataType>("dtype");
      DimVector dim_vec;
      dim_vec.push_back(ctx->Attr<int64_t>("range_shape"));  // Push range shape to DimVector
      *ctx->Dtype4ArgNameAndIndex("out", 0) = dtype;
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();  // clear batch axis
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // Still in process
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
