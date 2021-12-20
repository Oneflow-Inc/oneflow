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
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel());
  }

  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();

  // P2B
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_distribution->add_sbp_parallel()->mutable_partial_sum_parallel();
    out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> _ncclLogicalAllReduceOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel());
  }

  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();

  // P2S
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_distribution->add_sbp_parallel()->mutable_partial_sum_parallel();
    out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> _ncclLogicalReduceScatterOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), 0);
  }

  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();

  // S(0)->B
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> _ncclLogicalAllGatherOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
  CHECK_GE_OR_RETURN(in_split_axis, 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
  }

  cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();

  // S(1)->(B)
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
    out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferDataType(
    user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> _ncclLogicalAllGatherNoncontinuousOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalS2sOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
  const int64_t out_split_axis = ctx->user_op_conf().attr<int64_t>("out_split_axis");
  const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
  }

  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();

  // S(in)->S(out)
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
    out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> _ncclLogicalS2sOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

}  // namespace oneflow
