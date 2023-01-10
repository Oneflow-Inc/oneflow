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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedReluDropoutGrad(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("dy", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4FusedReluDropoutGrad(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedReluDropoutGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedReluDropoutGrad(ctx);
}

/*static*/ Maybe<void> FusedReluDropoutGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedReluDropoutGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedReluDropoutGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4FusedReluDropoutGrad(ctx);
}

}  // namespace oneflow
