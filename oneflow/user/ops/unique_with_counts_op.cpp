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

/*static*/ Maybe<void> UniqueWithCountsOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> UniqueWithCountsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  CHECK_EQ_OR_RETURN(x.shape().NumAxes(), 1);

  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  *y->mut_shape() = x.shape();
  *y->mut_is_dynamic() = x.is_dynamic();

  user_op::TensorDesc* idx = ctx->MutOutputTensorDesc("idx", 0);
  *idx->mut_shape() = x.shape();
  *idx->mut_is_dynamic() = x.is_dynamic();

  user_op::TensorDesc* count = ctx->MutOutputTensorDesc("count", 0);
  *count->mut_shape() = x.shape();
  *count->mut_is_dynamic() = x.is_dynamic();

  user_op::TensorDesc* num_unique = ctx->MutOutputTensorDesc("num_unique", 0);
  *num_unique->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UniqueWithCountsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UniqueWithCountsOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  auto out_idx = ctx->Attr<DataType>("out_idx");
  CHECK_OR_RETURN(IsIndexDataType(out_idx));
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  *y->mut_data_type() = x.data_type();

  user_op::TensorDesc* idx = ctx->MutOutputTensorDesc("idx", 0);
  *idx->mut_data_type() = out_idx;

  user_op::TensorDesc* count = ctx->MutOutputTensorDesc("count", 0);
  *count->mut_data_type() = out_idx;
  user_op::TensorDesc* num_unique = ctx->MutOutputTensorDesc("num_unique", 0);
  *num_unique->mut_data_type() = out_idx;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
