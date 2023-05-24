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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> NonContiguousBinaryOp::GetSbp(user_op::SbpContext* ctx) {
  // only support broadcast
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("lhs", 0))
      .Broadcast(user_op::OpArg("rhs", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NonContiguousBinaryOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& lhs = ctx->InputShape("lhs", 0);
  const Shape& rhs = ctx->InputShape("rhs", 0);
  CHECK_EQ(lhs.NumAxes(), rhs.NumAxes());
  for (int i = 0; i < lhs.NumAxes(); i++) CHECK_EQ(lhs.At(i), rhs.At(i));
  ctx->SetOutputShape("y", 0, lhs);
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace) {
    ctx->SetOutputStride("y", 0, ctx->InputStride("lhs", 0));
  } else {  // set contiguous for y if not inplace
    ctx->SetOutputStride("y", 0, Stride(lhs));
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> NonContiguousBinaryOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> NonContiguousBinaryOp::InferDataType(user_op::InferContext* ctx) {
  auto lhs = ctx->InputDType("lhs", 0);
  auto rhs = ctx->InputDType("rhs", 0);
  ctx->SetOutputDType("y", 0, GetSizeOfDataType(lhs) >= GetSizeOfDataType(rhs) ? lhs : rhs);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NonContiguousBinaryOpGrad::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("lhs", 0))
      .Broadcast(user_op::OpArg("rhs", 0))
      .Broadcast(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("dlhs", 0))
      .Broadcast(user_op::OpArg("drhs", 0))
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NonContiguousBinaryOpGrad::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& lhs = ctx->InputShape("lhs", 0);
  const Shape& rhs = ctx->InputShape("rhs", 0);
  CHECK_EQ(lhs.NumAxes(), rhs.NumAxes());
  for (int i = 0; i < lhs.NumAxes(); i++) CHECK_EQ(lhs.At(i), rhs.At(i));
  ctx->SetOutputShape("dlhs", 0, lhs);
  ctx->SetOutputStride("dlhs", 0, ctx->InputStride("lhs", 0));
  ctx->SetOutputShape("drhs", 0, rhs);
  ctx->SetOutputStride("drhs", 0, ctx->InputStride("rhs", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> NonContiguousBinaryOpGrad::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> NonContiguousBinaryOpGrad::InferDataType(user_op::InferContext* ctx) {
  auto lhs = ctx->InputDType("lhs", 0);
  auto rhs = ctx->InputDType("rhs", 0);
  ctx->SetOutputDType("dlhs", 0, lhs);
  ctx->SetOutputDType("drhs", 0, rhs);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
