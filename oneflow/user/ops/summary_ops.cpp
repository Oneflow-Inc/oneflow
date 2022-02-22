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

namespace {

Maybe<void> CheckStepShape(const Shape* step) {
  CHECK_OR_RETURN(step->elem_cnt() == 1);
  return Maybe<void>::Ok();
}

Maybe<void> CheckStepShapeInCtx(user_op::InferContext* ctx) {
  JUST(CheckStepShape(&ctx->InputShape("step", 0)));
  return Maybe<void>::Ok();
}

Maybe<void> CheckInAndStepScalar(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& step_shape = ctx->InputShape("step", 0);
  CHECK_OR_RETURN(in_shape.elem_cnt() == 1 && step_shape.elem_cnt() == 1);
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> CreateSummaryWriterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> CreateSummaryWriterOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CreateSummaryWriterOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CreateSummaryWriterOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FlushSummaryWriterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> FlushSummaryWriterOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> FlushSummaryWriterOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> FlushSummaryWriterOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SummaryWriteScalarOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> SummaryWriteScalarOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return CheckInAndStepScalar(ctx);
}
/*static*/ Maybe<void> SummaryWriteScalarOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SummaryWriteScalarOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SummaryWriteHistogramOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> SummaryWriteHistogramOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return CheckStepShapeInCtx(ctx);
}
/*static*/ Maybe<void> SummaryWriteHistogramOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SummaryWriteHistogramOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SummaryWritePbOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> SummaryWritePbOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return CheckStepShapeInCtx(ctx);
}
/*static*/ Maybe<void> SummaryWritePbOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SummaryWritePbOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SummaryWriteImageOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> SummaryWriteImageOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return CheckStepShapeInCtx(ctx);
}
/*static*/ Maybe<void> SummaryWriteImageOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SummaryWriteImageOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

}  // namespace oneflow
