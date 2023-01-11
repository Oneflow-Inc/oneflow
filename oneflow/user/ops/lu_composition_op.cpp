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

/*static*/ Maybe<void> LUDecompositionOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& x_desc = ctx->InputTensorDesc("x", 0);
  auto x_shape = x_desc.shape();
  ctx->SetOutputShape("pivot", 0, Shape(x_shape.begin(), x_shape.end() - 1));
  ctx->SetOutputShape("LU", 0, x_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LUDecompositionOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> LUDecompositionOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes() - 2) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("LU", 0), i)
        .Split(user_op::OpArg("pivot", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LUDecompositionOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("LU", 0, ctx->InputDType("x", 0));
  ctx->SetOutputDType("pivot", 0, DataType::kInt32);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
