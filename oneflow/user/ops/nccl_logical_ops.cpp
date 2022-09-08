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
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // P->B
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllReduceOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalAllReduceOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // P->S(0)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_split_parallel());
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalReduceScatterOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // S(0)->B
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_split_parallel());
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalAllGatherOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // S(>=1)->B
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_split_parallel());
  CHECK_GE_OR_RETURN(input_nd_sbp->sbp_parallel(0).split_parallel().axis(), 1);
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalAllGatherNoncontinuousOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalAllGatherNoncontinuousOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterNoncontinuousOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterNoncontinuousOp::GetSbp(
    user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalReduceScatterNoncontinuousOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // P->S(0)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1) << "input_nd_sbp should be 1d.";
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1) << "output_nd_sbp should be 1d.";
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_partial_sum_parallel())
      << "input_nd_sbp should be partial_sum_parallel.";
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_split_parallel())
      << "output_nd_sbp should be split parallel.";
  CHECK_GE_OR_RETURN(output_nd_sbp->sbp_parallel(0).split_parallel().axis(), 1)
      << "output_nd_sbp split axis should greater equal 1.";
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1) << "parallel_hierarchy should be 1d.";

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalReduceScatterNoncontinuousOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalReduceScatterNoncontinuousOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalS2sOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // S->S
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_split_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_split_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 1);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalS2sOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalS2sOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogicalSendRecvOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalSendRecvOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalSendRecvOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_nd_sbp", output_nd_sbp));

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalSendRecvOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalSendRecvOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
