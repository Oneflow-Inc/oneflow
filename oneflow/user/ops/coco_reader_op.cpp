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

/* static */ Maybe<void> COCOReaderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  user_op::TensorDesc* image_desc = ctx->MutOutputTensorDesc("image", 0);
  image_desc->set_shape(Shape({batch_size}));
  user_op::TensorDesc* image_id_desc = ctx->MutOutputTensorDesc("image_id", 0);
  image_id_desc->set_shape(Shape({batch_size}));
  user_op::TensorDesc* image_size_desc = ctx->MutOutputTensorDesc("image_size", 0);
  image_size_desc->set_shape(Shape({batch_size, 2}));
  user_op::TensorDesc* bbox_desc = ctx->MutOutputTensorDesc("gt_bbox", 0);
  bbox_desc->set_shape(Shape({batch_size}));
  user_op::TensorDesc* label_desc = ctx->MutOutputTensorDesc("gt_label", 0);
  label_desc->set_shape(Shape({batch_size}));
  user_op::TensorDesc* segm_desc = ctx->MutOutputTensorDesc("gt_segm", 0);
  segm_desc->set_shape(Shape({batch_size}));
  user_op::TensorDesc* segm_index_desc = ctx->MutOutputTensorDesc("gt_segm_index", 0);
  segm_index_desc->set_shape(Shape({batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> COCOReaderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("image", 0);
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("image_id", 0));
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("image_size", 0));
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("gt_bbox", 0));
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("gt_label", 0));
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("gt_segm", 0));
  CHECK_OR_RETURN(nd_sbp == ctx->NdSbp4ArgNameAndIndex("gt_segm_index", 0));

  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  int64_t device_batch_size = batch_size;
  if (parallel_num > 1) {
    int64_t split_num = 1;
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) { split_num *= hierarchy.At(i); }
    }
    CHECK_EQ_OR_RETURN(device_batch_size % split_num, 0);
    device_batch_size /= split_num;
  }

  user_op::TensorDesc* image_desc = ctx->MutOutputTensorDesc("image", 0);
  image_desc->set_shape(Shape({device_batch_size}));
  user_op::TensorDesc* image_id_desc = ctx->MutOutputTensorDesc("image_id", 0);
  image_id_desc->set_shape(Shape({device_batch_size}));
  user_op::TensorDesc* image_size_desc = ctx->MutOutputTensorDesc("image_size", 0);
  image_size_desc->set_shape(Shape({device_batch_size, 2}));
  user_op::TensorDesc* bbox_desc = ctx->MutOutputTensorDesc("gt_bbox", 0);
  bbox_desc->set_shape(Shape({device_batch_size}));
  user_op::TensorDesc* label_desc = ctx->MutOutputTensorDesc("gt_label", 0);
  label_desc->set_shape(Shape({device_batch_size}));
  user_op::TensorDesc* segm_desc = ctx->MutOutputTensorDesc("gt_segm", 0);
  segm_desc->set_shape(Shape({device_batch_size}));
  user_op::TensorDesc* segm_index_desc = ctx->MutOutputTensorDesc("gt_segm_index", 0);
  segm_index_desc->set_shape(Shape({device_batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> COCOReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> COCOReaderOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::OutputArgModifier* image_modifier = GetOutputArgModifierFn("image", 0);
  CHECK_OR_RETURN(image_modifier != nullptr);
  image_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* image_id_modifier = GetOutputArgModifierFn("image_id", 0);
  CHECK_OR_RETURN(image_id_modifier != nullptr);
  image_id_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* image_size_modifier = GetOutputArgModifierFn("image_size", 0);
  CHECK_OR_RETURN(image_size_modifier != nullptr);
  image_size_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* gt_bbox_modifier = GetOutputArgModifierFn("gt_bbox", 0);
  CHECK_OR_RETURN(gt_bbox_modifier != nullptr);
  gt_bbox_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* gt_label_modifier = GetOutputArgModifierFn("gt_label", 0);
  CHECK_OR_RETURN(gt_label_modifier != nullptr);
  gt_label_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* gt_segm_modifier = GetOutputArgModifierFn("gt_segm", 0);
  CHECK_OR_RETURN(gt_segm_modifier != nullptr);
  gt_segm_modifier->set_header_infered_before_compute(false);

  user_op::OutputArgModifier* gt_segm_index_modifier = GetOutputArgModifierFn("gt_segm_index", 0);
  CHECK_OR_RETURN(gt_segm_index_modifier != nullptr);
  gt_segm_index_modifier->set_header_infered_before_compute(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> COCOReaderOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_split_parallel()->set_axis(0);
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> COCOReaderOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* image_desc = ctx->MutOutputTensorDesc("image", 0);
  image_desc->set_data_type(DataType::kTensorBuffer);
  user_op::TensorDesc* image_id_desc = ctx->MutOutputTensorDesc("image_id", 0);
  image_id_desc->set_data_type(DataType::kInt64);
  user_op::TensorDesc* image_size_desc = ctx->MutOutputTensorDesc("image_size", 0);
  image_size_desc->set_data_type(DataType::kInt32);
  user_op::TensorDesc* bbox_desc = ctx->MutOutputTensorDesc("gt_bbox", 0);
  bbox_desc->set_data_type(DataType::kTensorBuffer);
  user_op::TensorDesc* label_desc = ctx->MutOutputTensorDesc("gt_label", 0);
  label_desc->set_data_type(DataType::kTensorBuffer);
  user_op::TensorDesc* segm_desc = ctx->MutOutputTensorDesc("gt_segm", 0);
  segm_desc->set_data_type(DataType::kTensorBuffer);
  user_op::TensorDesc* segm_index_desc = ctx->MutOutputTensorDesc("gt_segm_index", 0);
  segm_index_desc->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
