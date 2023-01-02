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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> PadOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    if (padding_before[i] == 0 && padding_after[i] == 0) {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PadOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
  CHECK_EQ_OR_RETURN(padding_before.size(), x_shape.NumAxes());
  CHECK_EQ_OR_RETURN(padding_after.size(), x_shape.NumAxes());
  DimVector y_dim_vec(x_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
    y_dim_vec[i] = x_shape.At(i) + padding_before[i] + padding_after[i];
  }
  ctx->SetOutputShape("y", 0, Shape(y_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PadOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return PadOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> PadOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
