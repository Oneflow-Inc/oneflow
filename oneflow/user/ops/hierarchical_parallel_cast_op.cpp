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

/* static */ Maybe<void> HierarchicalParallelCastOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> HierarchicalParallelCastOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> HierarchicalParallelCastOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> HierarchicalParallelCastOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  const auto& conf = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");
  CHECK_EQ_OR_RETURN(conf.size(), parallel_hierarchy.NumAxes());
  for (const std::string& sbp_str : conf) {
    SbpParallel sbp_parallel;
    CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
    *in_distribution->add_sbp_parallel() = sbp_parallel;
    *out_distribution->add_sbp_parallel() = sbp_parallel;
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> HierarchicalParallelCastOp::GetNdSbpSignatureList(
    user_op::GetNdSbpSignatureListContext* ctx) {
  const auto& conf = ctx->Attr<std::vector<std::string>>("nd_sbp");
  NdSbpSignature nd_sbp_signature;
  for (const std::string& sbp_str : conf) {
    SbpParallel sbp_parallel;
    CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
    *(*nd_sbp_signature.mutable_bn_in_op2nd_sbp())[GenRepeatedBn("in", 0)].add_sbp_parallel() =
        sbp_parallel;
    *(*nd_sbp_signature.mutable_bn_in_op2nd_sbp())[GenRepeatedBn("out", 0)].add_sbp_parallel() =
        sbp_parallel;
  }
  ctx->AddNdSbpSignature(nd_sbp_signature);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> HierarchicalParallelCastOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> HierarchicalParallelCastLikeOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> HierarchicalParallelCastLikeOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> HierarchicalParallelCastLikeOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> HierarchicalParallelCastLikeOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  NdSbp* like_distribution = ctx->NdSbp4ArgNameAndIndex("like", 0);
  const NdSbp& hint_distribution = ctx->NdSbpHint4InputArgNameAndIndex("like", 0);
  *in_distribution = hint_distribution;
  *out_distribution = hint_distribution;
  *like_distribution = hint_distribution;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> HierarchicalParallelCastLikeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
