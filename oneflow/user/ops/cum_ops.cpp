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

Maybe<void> CumsumOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("y", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> CumsumOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> CumsumOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  auto dim = ctx->Attr<int64_t>("dim");
  for (auto i = dim + 1; i < in_tensor_desc.shape().NumAxes(); i++) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> CumsumOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> CumProdOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("y", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> CumProdOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> CumProdOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  auto dim = ctx->Attr<int64_t>("dim");
  for (auto i = dim + 1; i < in_tensor_desc.shape().NumAxes(); i++) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> CumProdOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> CumProdGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("dy", 0));
  return Maybe<void>::Ok();
}

Maybe<void> CumProdGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> CumProdGradOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& dy_tensor_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  auto dim = ctx->Attr<int64_t>("dim");
  for (auto i = dim + 1; i < dy_tensor_desc.shape().NumAxes(); i++) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("output", 0), i)
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> CumProdGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
