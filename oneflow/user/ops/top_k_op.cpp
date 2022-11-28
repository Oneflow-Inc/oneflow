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

/*static*/ Maybe<void> TopKOp::GetSbp(user_op::SbpContext* ctx) {
  // The current implementation can only do top_k in the last dimension and should use Broadcast
  // (by default) instead of Split for that dimension
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TopKOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape out_shape = ctx->InputShape("in", 0);
  out_shape.Set(
      out_shape.NumAxes() - 1,
      std::min(ctx->Attr<int32_t>("k"), static_cast<int32_t>(out_shape.dim_vec().back())));
  ctx->SetOutputShape("out", 0, out_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TopKOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TopKOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
