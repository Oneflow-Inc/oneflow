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

bool CheckBroadCastAble(const Shape& shape, const Shape& broadcast_shape) {
  int left_pad = broadcast_shape.size() - shape.size();
  if (left_pad < 0) { return false; }
  for (int i = 0; i < shape.size(); ++i) {
    int j = i + left_pad;
    if (shape[i] != 1 && shape[i] != broadcast_shape[j]) { return false; }
  }
  return true;
}

}  // namespace

/*static*/ Maybe<void> BroadCastInplaceGreaterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& x_desc = ctx->InputTensorDesc("x", 0);
  const auto& y_desc = ctx->InputTensorDesc("y", 0);
  auto x_shape = x_desc.shape();
  auto y_shape = y_desc.shape();
  bool broadcast_status = CheckBroadCastAble(y_shape, x_shape);
  CHECK_OR_RETURN(broadcast_status);
  ctx->SetOutputShape("out", 0, x_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadCastInplaceGreaterOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return BroadCastInplaceGreaterOp::InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> BroadCastInplaceGreaterOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadCastInplaceGreaterOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ScalarLogicalInplaceGreaterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ScalarLogicalInplaceGreaterOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return ScalarLogicalInplaceGreaterOp::InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> ScalarLogicalInplaceGreaterOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ScalarLogicalInplaceGreaterOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
