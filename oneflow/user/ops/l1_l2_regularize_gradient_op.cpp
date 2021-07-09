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

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape());
  *ctx->OutputShape("out", 0) = ctx->InputShape("model", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("model", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_NO_GRAD_USER_OP("l1_l2_regularize_gradient")
    .Input("model")
    .Input("model_diff")
    .Output("out")
    .Attr<float>("l1", 0)
    .Attr<float>("l2", 0)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(GetSbpSignatures)
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
      const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
      CHECK_EQ_OR_RETURN(model_diff.data_type(), model.data_type());
      *ctx->OutputDType("out", 0) = ctx->InputDType("model", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
