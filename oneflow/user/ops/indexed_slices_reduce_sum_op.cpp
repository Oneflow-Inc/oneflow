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

/* static */ Maybe<void> IndexedSlicesReduceSumOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_indices = ctx->InputTensorDesc("x_indices", 0);
  const user_op::TensorDesc& x_values = ctx->InputTensorDesc("x_values", 0);
  CHECK_LT_OR_RETURN(x_indices.shape().NumAxes(), x_values.shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, x_indices.shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(x_indices.shape().At(i), x_values.shape().At(i));
  }

  const int64_t n = x_indices.shape().elem_cnt();
  const int64_t m = x_values.shape().elem_cnt() / n;
  user_op::TensorDesc* y_indices = ctx->MutOutputTensorDesc("y_indices", 0);
  user_op::TensorDesc* y_values = ctx->MutOutputTensorDesc("y_values", 0);
  *y_indices = x_indices;
  y_indices->set_shape(Shape({n}));
  *y_values = x_values;
  y_values->set_shape(Shape({n, m}));
  user_op::TensorDesc* num_unique = ctx->MutOutputTensorDesc("num_unique", 0);
  num_unique->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> IndexedSlicesReduceSumOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> IndexedSlicesReduceSumOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> IndexedSlicesReduceSumOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_indices = ctx->InputTensorDesc("x_indices", 0);
  CHECK_OR_RETURN(IsIndexDataType(x_indices.data_type()));
  user_op::TensorDesc* num_unique = ctx->MutOutputTensorDesc("num_unique", 0);
  num_unique->set_data_type(DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
