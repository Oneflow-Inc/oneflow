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

/*static*/ Maybe<void> OnesLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& like_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0);
  FOR_RANGE(int64_t, i, 0, like_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("like", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("like", 0))
      .Broadcast(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> OnesLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("like", 0));
  ctx->SetOutputStride("out", 0, ctx->InputStride("like", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> OnesLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return OnesLikeOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> OnesLikeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("like", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> OnesLikeOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_sbp = ctx->NdSbpHint4InputArgNameAndIndex("like", 0);
  NdSbp* like_distribution = ctx->NdSbp4ArgNameAndIndex("like", 0);
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  *like_distribution = in_sbp;
  *out_distribution = in_sbp;
  for (auto& sbp : *out_distribution->mutable_sbp_parallel()) {
    if (sbp.has_partial_sum_parallel()) {
      sbp.Clear();
      *sbp.mutable_broadcast_parallel() = BroadcastParallel();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
