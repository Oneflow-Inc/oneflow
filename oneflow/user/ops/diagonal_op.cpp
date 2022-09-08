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

/* static */ Maybe<void> DiagonalOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const int32_t offset = ctx->Attr<int32_t>("offset");
  const ShapeView& in_shape = in.shape();
  const int32_t in_dim = in_shape.NumAxes();
  CHECK_GE_OR_RETURN(in_dim, 2);

  DimVector out_dim_vec = {};
  FOR_RANGE(int32_t, index, 2, in_dim) { out_dim_vec.push_back(in_shape.At(index)); }
  int32_t last_dim = 0;
  if (offset >= 0) {
    last_dim = std::min(in_shape.At(0), in_shape.At(1) - offset);
  } else {
    last_dim = std::min(in_shape.At(0) + offset, in_shape.At(1));
  }
  if (last_dim < 0) { last_dim = 0; }
  out_dim_vec.push_back(last_dim);

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(false);
  out_desc->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DiagonalOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DiagonalOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagonalOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagonalGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const Shape& in_shape = in.shape();
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_shape(Shape(in_shape.dim_vec()));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DiagonalGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DiagonalGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagonalGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
