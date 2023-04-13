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

/* static */ Maybe<void> RocAucScoreOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  const Shape& pred_shape = ctx->InputTensorDesc("pred", 0).shape();
  const Shape& label_shape = ctx->InputTensorDesc("label", 0).shape();
  CHECK_EQ_OR_RETURN(pred_shape.elem_cnt(), label_shape.elem_cnt())
      << "pred and label MUST have same element count.";
  out_desc->set_is_dynamic(false);
  out_desc->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RocAucScoreOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RocAucScoreOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> RocAucScoreOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kFloat);
  const user_op::TensorDesc& label = ctx->InputTensorDesc("label", 0);
  CHECK_OR_RETURN(IsFloatingDataType(label.data_type()) || IsIntegralDataType(label.data_type()))
      << "Input `label` data type " << DataType_Name(label.data_type()) << " is not supported.";
  const user_op::TensorDesc& pred = ctx->InputTensorDesc("pred", 0);
  CHECK_OR_RETURN(pred.data_type() == DataType::kFloat)
      << "Input `pred` data type " << DataType_Name(pred.data_type()) << " is not supported.";
  return Maybe<void>::Ok();
}

}  // namespace oneflow
