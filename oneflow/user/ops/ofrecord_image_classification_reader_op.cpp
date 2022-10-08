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

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  user_op::TensorDesc* image_tensor = ctx->MutOutputTensorDesc("image", 0);
  user_op::TensorDesc* label_tensor = ctx->MutOutputTensorDesc("label", 0);
  int32_t batch_size = ctx->Attr<int32_t>("batch_size");
  image_tensor->set_shape(Shape({batch_size}));
  label_tensor->set_shape(Shape({batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  user_op::TensorDesc* image_tensor = ctx->MutOutputTensorDesc("image", 0);
  user_op::TensorDesc* label_tensor = ctx->MutOutputTensorDesc("label", 0);
  int32_t local_batch_size = ctx->Attr<int32_t>("batch_size");
  int64_t parallel_num = ctx->parallel_ctx().parallel_num();

  if (parallel_num > 1) {
    int64_t split_num = 1;
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("image", 0);
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) { split_num *= hierarchy.At(i); }
    }
    CHECK_EQ_OR_RETURN(local_batch_size % split_num, 0);
    local_batch_size /= split_num;
  }
  image_tensor->set_shape(Shape({local_batch_size}));
  label_tensor->set_shape(Shape({local_batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::OutputArgModifier* image_modifier = GetOutputArgModifierFn("image", 0);
  CHECK_OR_RETURN(image_modifier != nullptr);
  image_modifier->set_header_infered_before_compute(false);
  user_op::OutputArgModifier* label_modifier = GetOutputArgModifierFn("label", 0);
  CHECK_OR_RETURN(label_modifier != nullptr);
  label_modifier->set_header_infered_before_compute(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("image", 0, DataType::kTensorBuffer);
  ctx->SetOutputDType("label", 0, DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
