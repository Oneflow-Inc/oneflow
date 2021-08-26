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

REGISTER_USER_OP("dot") // #1 注册OP
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      CHECK_OR_RETURN(x.shape() == y.shape()) << "Input tensor shape is different";
      CHECK_OR_RETURN(x.shape().NumAxes() == 1) << "Input tensor is not 1D";
      *ctx->OutputShape("out", 0) = Shape({1});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .PartialSum(user_op::OpArg("out", 0))
          .Build();

      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      CHECK_OR_RETURN(x.data_type() == y.data_type()) << "The input tensor type is different";
      *ctx->OutputDType("out", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });
}  // namespace

}  // namespace oneflow
