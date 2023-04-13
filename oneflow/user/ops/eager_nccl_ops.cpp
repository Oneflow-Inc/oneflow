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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

/* static */ Maybe<void> EagerCclAllReduceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerCclAllReduceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerCclAllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(user_op::OpArg("in", 0)).Broadcast(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclAllReduceOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerCclAllReduceOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerCclBroadcastOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  size_t size = ctx->input_size("in");
  const std::vector<Shape>& shape_list = ctx->Attr<std::vector<Shape>>("shape_list");
  CHECK_EQ_OR_RETURN(size, ctx->output_size("out"))
      << "the size of input tensor tuple should equal the size of output tensor tuple.";
  for (int i = 0; i < size; ++i) { ctx->SetOutputShape("out", i, JUST(VectorAt(shape_list, i))); }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerCclBroadcastOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerCclBroadcastOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  ctx->NewBuilder().Split(ctx->inputs(), 0).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclBroadcastOp::InferDataType(user_op::InferContext* ctx) {
  size_t size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(size, ctx->output_size("out"))
      << "the size of input tensor tuple should equal the size of output tensor tuple.";
  for (int i = 0; i < size; ++i) { ctx->SetOutputDType("out", i, ctx->InputDType("in", i)); }
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerCclBroadcastOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerNcclTouchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclTouchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclTouchOp::GetSbp(user_op::SbpContext* ctx) {
  // local only
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclTouchOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerNcclTouchOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerCclReduceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerCclReduceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerCclReduceOp::GetSbp(user_op::SbpContext* ctx) {
  UNIMPLEMENTED_THEN_RETURN() << "global tensor are not supported";
}

/* static */ Maybe<void> EagerCclReduceOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerCclReduceOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerCclReduceScatterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclReduceScatterOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& input_shape = ctx->InputShape("in", 0);
  const auto& shape = ctx->Attr<Shape>("output_shape");
  Symbol<ParallelDesc> parallel_desc =
      JUST(TxtStringToPlacement(ctx->Attr<std::string>("parallel_conf")));
  CHECK_EQ_OR_RETURN(input_shape.elem_cnt(), shape.elem_cnt() * parallel_desc->parallel_num())
      << Error::RuntimeError()
      << "output tensor size must be equal to world_size times input tensor size";
  CHECK_EQ_OR_RETURN(ctx->InputDType("in", 0), ctx->Attr<DataType>("output_dtype"))
      << Error::RuntimeError() << "output tensor must have the same type as input tensor";
  ctx->SetOutputShape("out", 0, ctx->Attr<Shape>("output_shape"));
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("output_dtype"));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclReduceScatterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerCclReduceScatterOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel() || sbp_hint.has_broadcast_parallel());
  }
  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // P2S or B2S
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  in_nd_sbp->CopyFrom(in_dis_hint);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclReduceScatterOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerCclReduceScatterOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerCclAllGatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerCclAllGatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const auto& input_shape = ctx->InputShape("in", 0);
  const auto& shape = ctx->Attr<Shape>("output_shape");
  Symbol<ParallelDesc> parallel_desc =
      JUST(TxtStringToPlacement(ctx->Attr<std::string>("parallel_conf")));
  CHECK_EQ_OR_RETURN(input_shape.elem_cnt() * parallel_desc->parallel_num(), shape.elem_cnt())
      << Error::RuntimeError()
      << "output tensor size must be equal to world_size times input tensor size";
  CHECK_EQ_OR_RETURN(ctx->InputDType("in", 0), ctx->Attr<DataType>("output_dtype"))
      << Error::RuntimeError() << Error::RuntimeError()
      << "output tensor must have the same type as input tensor";
  ctx->SetOutputShape("out", 0, ctx->Attr<Shape>("output_shape"));
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("output_dtype"));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclAllGatherOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerCclAllGatherOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), 0);
  }

  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // S(0)->B
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    out_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerCclAllGatherOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerCclAllGatherOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

/* static */ Maybe<void> EagerNcclS2sOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclS2sOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerNcclS2sOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
  const int64_t out_split_axis = ctx->user_op_conf().attr<int64_t>("out_split_axis");
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
  }

  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // S(in)->S(out)
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
    out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclS2sOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerNcclS2sOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
