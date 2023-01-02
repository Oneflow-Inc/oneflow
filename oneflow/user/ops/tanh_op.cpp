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

/*static*/ Maybe<void> TanhOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> TanhOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> TanhOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TanhOp::InferDataType(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);
}

/*static*/ Maybe<void> TanhGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> TanhGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> TanhGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TanhGradOp::InferDataType(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);
}

}  // namespace oneflow
