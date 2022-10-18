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

/* static */ Maybe<void> MultinomialWithReplacementOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  int32_t num_samples = ctx->Attr<int32_t>("num_samples");
  const Shape& x_shape = ctx->InputShape("x", 0);
  if (x_shape.NumAxes() == 1) {
    ctx->SetOutputShape("out", 0, Shape({num_samples}));
  } else {
    ctx->SetOutputShape("out", 0, Shape({x_shape.At(0), num_samples}));
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MultinomialWithReplacementOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultinomialWithReplacementOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  if (x_shape.NumAxes() == 2) {
    ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultinomialWithReplacementOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
