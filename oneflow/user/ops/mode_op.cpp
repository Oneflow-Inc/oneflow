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

/*static*/ Maybe<void> ModeOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  int64_t num_axes = in_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  if (num_axes == 0) {
    ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ModeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& reduce_shape = CreateReducedShape(input_shape, {-1});
  ctx->SetOutputShape("values", 0, reduce_shape.RemoveOnes({-1}));
  ctx->SetOutputShape("indices", 0, reduce_shape.RemoveOnes({-1}));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ModeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ModeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("values", 0, ctx->InputDType("input", 0));
  ctx->SetOutputDType("indices", 0, DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
