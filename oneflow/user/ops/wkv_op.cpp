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
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {

/*static*/ Maybe<void> WkvOp::GetSbp(user_op::SbpContext* ctx) {
  // return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("w", 0))
      .Broadcast(user_op::OpArg("u", 0))
      .Split(user_op::OpArg("k", 0), 0)
      .Split(user_op::OpArg("v", 0), 0)
      .Split(user_op::OpArg("y", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& v_shape = ctx->InputShape("v", 0);
  ctx->SetOutputShape("y", 0, v_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvOp::InferDataType(user_op::InferContext* ctx) {
  DataType v_dtype = ctx->InputDType("v", 0);
  ctx->SetOutputDType("y", 0, v_dtype);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WkvGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("w", 0))
      .Broadcast(user_op::OpArg("u", 0))
      .Split(user_op::OpArg("k", 0), 0)
      .Split(user_op::OpArg("v", 0), 0)
      .Split(user_op::OpArg("gy", 0), 0)
      .Broadcast(user_op::OpArg("gw", 0))
      .Broadcast(user_op::OpArg("gu", 0))
      .Split(user_op::OpArg("gk", 0), 0)
      .Split(user_op::OpArg("gv", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const int64_t B = ctx->Attr<int64_t>("B");
  const int64_t C = ctx->Attr<int64_t>("C");
  const Shape& gy_shape = ctx->InputShape("gy", 0);
  ctx->SetOutputShape("gw", 0, Shape({B, C}));
  ctx->SetOutputShape("gu", 0, Shape({B, C}));
  ctx->SetOutputShape("gk", 0, gy_shape);
  ctx->SetOutputShape("gv", 0, gy_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvGradOp::InferDataType(user_op::InferContext* ctx) {
  DataType gy_dtype = ctx->InputDType("gy", 0);
  ctx->SetOutputDType("gw", 0, gy_dtype);
  ctx->SetOutputDType("gu", 0, gy_dtype);
  ctx->SetOutputDType("gk", 0, gy_dtype);
  ctx->SetOutputDType("gv", 0, gy_dtype);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
