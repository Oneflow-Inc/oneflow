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

/* static */ Maybe<void> DiagOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const int32_t diagonal = ctx->Attr<int32_t>("diagonal");
  const ShapeView& in_shape = in.shape();
  const int32_t in_dim = in_shape.NumAxes();
  CHECK_GE_OR_RETURN(in_dim, 1);
  CHECK_LE_OR_RETURN(in_dim, 2);

  DimVector out_dim_vec = {0};
  if (in_dim == 1) {
    int32_t out_tensor_size = in_shape.At(0) + std::abs(diagonal);
    out_dim_vec[0] = out_tensor_size;
    out_dim_vec.emplace_back(out_tensor_size);
  } else {
    if (diagonal >= 0) {
      out_dim_vec[0] = std::min(in_shape.At(0), in_shape.At(1) - diagonal);
    } else {
      out_dim_vec[0] = std::min(in_shape.At(0) + diagonal, in_shape.At(1));
    }
    // For 0-size Tensor.
    CHECK_GE_OR_RETURN(out_dim_vec[0], 0);  // NOLINT
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(false);
  out_desc->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DiagOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DiagOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const Shape& in_shape = in.shape();
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_shape(Shape(in_shape.dim_vec()));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DiagGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DiagGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DiagGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
