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

/*static*/ Maybe<void> LinalgCrossOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("input", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LinalgCrossOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> LinalgCrossOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  const int64_t num_axes = input.shape().NumAxes();
  const int64_t dim = ctx->Attr<int64_t>("dim");

  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (i == dim) continue;
    ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("other", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LinalgCrossOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow