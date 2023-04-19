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

/* static */ Maybe<void> NormalTensorTensorOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NormalTensorTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& mean_shape = ctx->InputShape("mean", 0);
  const Shape& std_shape = ctx->InputShape("std", 0);
  CHECK_EQ_OR_RETURN(mean_shape.elem_cnt(), std_shape.elem_cnt());
  size_t dimsA = mean_shape.NumAxes();
  size_t dimsB = std_shape.NumAxes();
  const Shape& out_shape =  dimsA > dimsB ? mean_shape : std_shape;
  ctx->SetOutputShape("out", 0, out_shape);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("mean", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NormalTensorTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}


/* static */ Maybe<void> NormalTensorTensorOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("mean", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow


