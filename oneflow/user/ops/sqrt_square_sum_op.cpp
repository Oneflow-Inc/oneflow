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

/*static*/ Maybe<void> SqrtSquareSumOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t num_x_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_x_axes) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).PartialSum(user_op::OpArg("y", 0)).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SqrtSquareSumOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_shape(Shape({}));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SqrtSquareSumOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SqrtSquareSumOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
