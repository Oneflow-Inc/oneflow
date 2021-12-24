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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ auto FlipOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const int input_dims = x_desc.shape().NumAxes();
  const std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
  CHECK_OR_RETURN(dims.size() <= input_dims) << "len of dims must less than len of input tensor";
  for (auto x : dims) { CHECK_OR_RETURN(x < input_dims) << "dims parameter is illegal."; }
  user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
  *y_desc->mut_shape() = x_desc.shape();
  return Maybe<void>::Ok();
}
/*static*/ auto FlipOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return FlipOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FlipOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ auto FlipOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ auto FlipGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  Shape* dx_shape = ctx->OutputShape("dx", 0);
  *dx_shape = dy_shape;
  return Maybe<void>::Ok();
}
/*static*/ auto FlipGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return FlipGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FlipGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ auto FlipGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
