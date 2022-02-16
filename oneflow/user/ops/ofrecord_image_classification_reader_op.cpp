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
  user_op::TensorDesc* image_tensor = ctx->OutputTensorDesc("image", 0);
  user_op::TensorDesc* label_tensor = ctx->OutputTensorDesc("label", 0);
  int32_t batch_size = ctx->Attr<int32_t>("batch_size");
  *image_tensor->mut_shape() = Shape({batch_size});
  *label_tensor->mut_shape() = Shape({batch_size});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  user_op::TensorDesc* image_tensor = ctx->OutputTensorDesc("image", 0);
  user_op::TensorDesc* label_tensor = ctx->OutputTensorDesc("label", 0);
  int32_t local_batch_size = ctx->Attr<int32_t>("batch_size");
  const SbpParallel& sbp = ctx->SbpParallel4ArgNameAndIndex("image", 0);
  int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  if (sbp.has_split_parallel() && parallel_num > 1) {
    CHECK_EQ_OR_RETURN(local_batch_size % parallel_num, 0);
    local_batch_size /= parallel_num;
  }
  *image_tensor->mut_shape() = Shape({local_batch_size});
  *label_tensor->mut_shape() = Shape({local_batch_size});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageClassificationReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
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
  *ctx->OutputDType("image", 0) = DataType::kTensorBuffer;
  *ctx->OutputDType("label", 0) = DataType::kTensorBuffer;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
