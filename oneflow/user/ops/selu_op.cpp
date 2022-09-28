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

/*static*/ Maybe<void> SeluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SeluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SeluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SeluOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SeluGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SeluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(dy_shape == x_shape)
      << Error::RuntimeError() << "Tensors dy and x must be the same shape";
  ctx->SetOutputShape("dx", 0, dy_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SeluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SeluGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("dy", 0), ctx->InputDType("x", 0))
      << Error::TypeError() << "Tensors dy and x must have same type";
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
