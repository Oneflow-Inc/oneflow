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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> CastToTickOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CastToTickOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CastToTickOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> CastToTickOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), parallel_hierarchy.NumAxes());

  NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();
  // in use hint
  in_distribution->CopyFrom(in_dis_hint);

  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    // out dim1 = broadcast
    out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CastToTickOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
