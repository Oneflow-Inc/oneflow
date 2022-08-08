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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
      // TODO: infer shape by extracting Ops from mlir_assembly
      CHECK_EQ(ctx->inputs().size(), 2);
      CHECK_EQ(ctx->outputs().size(), 1);
      const Shape& in_shape = ctx->InputShape("in", 0);
      Shape* out_shape = ctx->MutOutputShape("out", 0);
      *out_shape = in_shape;
      *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 1);
      return Maybe<void>::Ok();
}

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> MlirJitOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::GetSbp(user_op::SbpContext* ctx) { return GetSbpFn(ctx); }

Maybe<void> MlirJitOp::InferDataType(user_op::InferContext* ctx) { return InferDataType(ctx); }

}  // namespace oneflow
