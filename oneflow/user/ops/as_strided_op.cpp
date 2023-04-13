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

/* static */ auto AsStridedOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const auto& size = ctx->Attr<std::vector<int64_t>>("size");
  const auto& stride = ctx->Attr<std::vector<int64_t>>("stride");
  CHECK_EQ_OR_RETURN(size.size(), stride.size()) << "mismatch in length of strides and shape";
  DimVector out_vec;
  out_vec.insert(out_vec.end(), size.cbegin(), size.cend());
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("output", 0);
  output_desc->set_shape(Shape(out_vec));
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return AsStridedOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto AsStridedOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

/* static */ auto AsStridedGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& input_shape = ctx->InputShape("input", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_shape(input_shape);
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return AsStridedGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto AsStridedGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
