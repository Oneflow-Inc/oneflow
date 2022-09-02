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
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/user/ops/nccl_logical_util.h"

namespace oneflow {

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllReduceOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllReduceOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // (*, P) -> (*, B)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 2);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 2);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0) == output_nd_sbp->sbp_parallel(0));
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(1).has_partial_sum_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(1).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 2);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllReduceOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogical_2DSameDim0AllReduceOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim1AllReduceOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim1AllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim1AllReduceOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // (P, *) -> (B, *)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 2);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 2);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(1) == output_nd_sbp->sbp_parallel(1));
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 2);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim1AllReduceOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogical_2DSameDim1AllReduceOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // (*, S(0)) -> (*, B)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 2);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 2);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0) == output_nd_sbp->sbp_parallel(0));
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(1).has_split_parallel());
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel(1).split_parallel().axis(), 0);
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(1).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 2);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogical_2DSameDim0AllGatherOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherNoncontinuousOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherNoncontinuousOp::GetSbp(
    user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherNoncontinuousOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // (*, S(>=1)) -> (*, B)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 2);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 2);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0) == output_nd_sbp->sbp_parallel(0));
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(1).has_split_parallel());
  CHECK_GE_OR_RETURN(input_nd_sbp->sbp_parallel(1).split_parallel().axis(), 1);
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(1).has_broadcast_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 2);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0AllGatherNoncontinuousOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>>
_ncclLogical_2DSameDim0AllGatherNoncontinuousOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0All2allOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0All2allOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0All2allOp::InferNdSbp(
    user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", output_nd_sbp));
  // (*, S) -> (*, S)
  CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), 2);
  CHECK_EQ_OR_RETURN(output_nd_sbp->sbp_parallel_size(), 2);
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(0) == output_nd_sbp->sbp_parallel(0));
  CHECK_OR_RETURN(input_nd_sbp->sbp_parallel(1).has_split_parallel());
  CHECK_OR_RETURN(output_nd_sbp->sbp_parallel(1).has_split_parallel());
  CHECK_EQ_OR_RETURN(ctx->parallel_hierarchy().NumAxes(), 2);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogical_2DSameDim0All2allOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogical_2DSameDim0All2allOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
