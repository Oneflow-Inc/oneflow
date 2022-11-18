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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> FwdInferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1_desc = ctx->InputTensorDesc("x1", 0);
  const user_op::TensorDesc& x2_desc = ctx->InputTensorDesc("x2", 0);
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("out", 0);

  int64_t ndim = x1_desc.shape().NumAxes();
  Shape output_shape(x1_desc.shape().begin(), x1_desc.shape().end() - 2);
  output_shape.emplace_back(x1_desc.shape().At(ndim - 2));
  output_shape.emplace_back(x2_desc.shape().At(ndim - 2));
  output_desc->set_shape(Shape(output_shape));

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CDistOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return FwdInferTensorDesc(ctx);
}

/*static*/ Maybe<void> CDistOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CDistOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> CDistOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1_desc = ctx->InputTensorDesc("x1", 0);
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("out", 0);
  output_desc->set_data_type(x1_desc.data_type());
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> BwdInferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1_desc = ctx->InputTensorDesc("x1", 0);
  const user_op::TensorDesc& x2_desc = ctx->InputTensorDesc("x2", 0);
  user_op::TensorDesc* dx1_desc = ctx->MutOutputTensorDesc("dx1", 0);
  user_op::TensorDesc* dx2_desc = ctx->MutOutputTensorDesc("dx2", 0);

  dx1_desc->set_shape(x1_desc.shape());
  dx2_desc->set_shape(x2_desc.shape());

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CDistGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return BwdInferTensorDesc(ctx);
}

/*static*/ Maybe<void> CDistGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CDistGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> CDistGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1_desc = ctx->InputTensorDesc("x1", 0);
  user_op::TensorDesc* dx1_desc = ctx->MutOutputTensorDesc("dx1", 0);
  user_op::TensorDesc* dx2_desc = ctx->MutOutputTensorDesc("dx2", 0);
  dx1_desc->set_data_type(x1_desc.data_type());
  dx2_desc->set_data_type(x1_desc.data_type());

  return Maybe<void>::Ok();
}

}  // namespace oneflow
