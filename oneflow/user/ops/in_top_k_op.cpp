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

/* static */ Maybe<void> InTopKOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& targets = ctx->InputTensorDesc("targets", 0);
  const user_op::TensorDesc& predictions = ctx->InputTensorDesc("predictions", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_EQ_OR_RETURN(targets.shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(predictions.shape().NumAxes(), 2);
  const bool is_dynamic = targets.is_dynamic();
  CHECK_EQ_OR_RETURN(is_dynamic, predictions.is_dynamic());
  out->set_is_dynamic(is_dynamic);
  out->set_shape(targets.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> InTopKOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> InTopKOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> InTopKOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& targets = ctx->InputTensorDesc("targets", 0);
  CHECK_OR_RETURN(IsIndexDataType(targets.data_type()));
  const user_op::TensorDesc& predictions = ctx->InputTensorDesc("predictions", 0);
  CHECK_EQ_OR_RETURN(predictions.data_type(), DataType::kFloat)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kFloat) << ", but got "
      << DataType_Name(predictions.data_type());
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(kBool);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
