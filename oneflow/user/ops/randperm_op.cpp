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

/*static*/ Maybe<void> RandpermOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_broadcast_parallel();
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}
/*static*/ Maybe<void> RandpermOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }
/*static*/ Maybe<void> RandpermOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  int32_t n = ctx->Attr<int32_t>("n");
  CHECK_GE_OR_RETURN(n, 0);
  *out_shape = Shape({n});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> RandpermOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> RandpermOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kInt32;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
