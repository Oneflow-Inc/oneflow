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

REGISTER_NO_GRAD_USER_OP("uniform_int")
    .Output("out")
    .SetOutputBufferNum(1)
    .Attr<int64_t>("from", 0)
    .Attr<int64_t>("to", 1)
    .Attr<int64_t>("seed")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .Attr<std::string>("nd_sbp")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->OutputShape("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      auto dtype = ctx->Attr<DataType>("dtype");
      *ctx->OutputDType("out", 0) = dtype;
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      cfg::SbpParallel default_sbp;
      default_sbp.mutable_broadcast_parallel();
      return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
    });

}  // namespace oneflow
