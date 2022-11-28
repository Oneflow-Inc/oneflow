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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ auto GenerateRandomBatchPermutationIndicesOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputShape("y", 0, Shape({ctx->InputShape("x", 0).At(0)}));
  return Maybe<void>::Ok();
}
/*static*/ auto GenerateRandomBatchPermutationIndicesOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return GenerateRandomBatchPermutationIndicesOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto GenerateRandomBatchPermutationIndicesOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  ctx->NewBuilder().PartialSum(user_op::OpArg("x", 0)).Broadcast(user_op::OpArg("y", 0)).Build();
  const auto& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Broadcast(user_op::OpArg("y", 0)).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ auto GenerateRandomBatchPermutationIndicesOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  ctx->SetOutputDType("y", 0, DataType::kInt32);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
