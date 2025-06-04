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

Maybe<void> InferNmsTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape({ctx->InputShape("in", 0).At(0)}));
  return Maybe<void>::Ok();
}

Maybe<void> InferNmsDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kInt8);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> NmsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferNmsTensorDesc(ctx);
}

/*static*/ Maybe<void> NmsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NmsOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> NmsOp::InferDataType(user_op::InferContext* ctx) {
  return InferNmsDataType(ctx);
}

}  // namespace oneflow
