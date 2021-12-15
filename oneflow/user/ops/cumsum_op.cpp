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

namespace {

REGISTER_USER_OP("cumsum")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("dim")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
