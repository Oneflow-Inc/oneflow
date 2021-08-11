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

REGISTER_USER_OP("prelu")
    .Input("x")
    .Input("alpha")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const Shape& alpha_shape = ctx->InputShape("alpha", 0);
      CHECK_EQ_OR_RETURN(alpha_shape.NumAxes(), 1);
      *y_desc->mut_shape() = x_desc.shape();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("prelu_grad")
    .Input("dy")
    .Input("x")
    .Input("alpha")
    .Output("dx")
    .Output("alpha_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
      const user_op::TensorDesc& alpha_desc = ctx->InputTensorDesc("alpha", 0);
      CHECK_EQ_OR_RETURN(alpha_desc.shape().NumAxes(), 1);
      CHECK_OR_RETURN((alpha_desc.shape().At(0) == x_desc.shape().At(1))
                      || (alpha_desc.shape().At(0) == 1));
      CHECK_EQ_OR_RETURN(dy_desc.shape(), x_desc.shape());
      CHECK_EQ_OR_RETURN(dy_desc.data_type(), x_desc.data_type());
      *dx_desc->mut_shape() = x_desc.shape();
      *ctx->OutputShape("alpha_diff", 0) = alpha_desc.shape();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
      *ctx->OutputDType("alpha_diff", 0) = ctx->InputDType("alpha", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
