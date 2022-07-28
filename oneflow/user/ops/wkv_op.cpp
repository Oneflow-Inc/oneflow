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

/*static*/ Maybe<void> WkvOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> WkvOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& v_shape = ctx->InputShape("v", 0);
  *ctx->MutOutputShape("y", 0) = v_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WkvOp::InferDataType(user_op::InferContext* ctx) {
  DataType v_dtype = ctx->InputDType("v", 0);
  *ctx->MutOutputDType("y", 0) = v_dtype;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WkvGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> WkvGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& gy_shape = ctx->InputShape("gy", 0);
  *ctx->MutOutputShape("y", 0) = gy_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WkvGradOp::InferDataType(user_op::InferContext* ctx) {
  DataType gy_dtype = ctx->InputDType("gy", 0);
  *ctx->MutOutputDType("y", 0) = gy_dtype;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
