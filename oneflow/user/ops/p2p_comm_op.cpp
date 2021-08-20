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

REGISTER_NO_GRAD_USER_OP("send")
    .Input("in")
    .Attr<int64_t>("dst_process_id")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // Do nothing.
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { UNIMPLEMENTED_THEN_RETURN(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // Do nothing.
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_USER_OP("recv")
    .Output("out")
    .Attr<int64_t>("src_process_id")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->Attr<Shape>("shape");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { UNIMPLEMENTED_THEN_RETURN(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
