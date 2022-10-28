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

Maybe<void> InferDataTypeLogicalNot(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, DataType::kBool);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> LogicalNotOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}

/*static*/ Maybe<void> LogicalNotOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LogicalNotOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}

/* static */ Maybe<void> LogicalNotOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataTypeLogicalNot(ctx);
}

}  // namespace oneflow
