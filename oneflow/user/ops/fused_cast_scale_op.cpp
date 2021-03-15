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

Maybe<void> TensorDescInfer(user_op::InferContext* ctx) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* scalar = ctx->TensorDesc4ArgNameAndIndex("scalar", 0);
  CHECK_EQ_OR_RETURN(scalar->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(scalar->shape().At(0), 1);
  user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  *y = *x;
  *y->mut_data_type() = scalar->data_type();
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const auto& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  for (int i = 0; i < x.shape().NumAxes(); ++i) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("scalar", 0))
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("scalar", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("scalar", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("fused_cast_scale")
    .Input("x")
    .Input("scalar")
    .Output("y")
    .SetTensorDescInferFn(TensorDescInfer)
    .SetGetSbpFn(GetSbpSignatures);

}  // namespace
}  // namespace oneflow
