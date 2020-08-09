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

namespace oneflow {

namespace {

Maybe<void> CheckOpAttr(const user_op::UserOpDefWrapper& def,
                        const user_op::UserOpConfWrapper& conf) {
  bool check_failed = false;
  std::ostringstream err;
  err << "Illegal attr value for " << conf.op_type_name() << " op, op_name: " << conf.op_name();

  const auto& min_iou_vec = conf.attr<std::vector<float>>("min_iou_overlaps");
  const auto& max_iou_vec = conf.attr<std::vector<float>>("max_iou_overlaps");
  bool invalid_iou = false;
  if (min_iou_vec.size() != max_iou_vec.size()) {
    invalid_iou = true;
  } else if (min_iou_vec.size() == 0) {
    invalid_iou = true;
  } else {
    FOR_RANGE(size_t, i, 0, min_iou_vec.size()) {
      if (min_iou_vec.at(i) > 1.0f || max_iou_vec.at(i) > 1.0f) { invalid_iou = true; }
      if (min_iou_vec.at(i) >= max_iou_vec.at(i) && max_iou_vec.at(i) >= 0) { invalid_iou = true; }
    }
  }
  if (invalid_iou) {
    err << ", min_iou_overlaps: {";
    std::copy(min_iou_vec.begin(), min_iou_vec.end() - 1, std::ostream_iterator<float>(err, ","));
    err << min_iou_vec.back() << "}";
    err << ", max_iou_overlaps: {";
    std::copy(max_iou_vec.begin(), max_iou_vec.end() - 1, std::ostream_iterator<float>(err, ","));
    err << max_iou_vec.back() << "}";
    check_failed = true;
  }

  float min_width_shrink_rate = conf.attr<float>("min_width_shrink_rate");
  float max_width_shrink_rate = conf.attr<float>("max_width_shrink_rate");
  float min_height_shrink_rate = conf.attr<float>("min_height_shrink_rate");
  float max_height_shrink_rate = conf.attr<float>("max_height_shrink_rate");
  if (min_width_shrink_rate <= 0.0f || min_height_shrink_rate <= 0.0f
      || min_width_shrink_rate > max_width_shrink_rate
      || min_height_shrink_rate > max_height_shrink_rate || max_width_shrink_rate > 1.0f
      || max_height_shrink_rate > 1.0f) {
    err << ", min_width_shrink_rate: " << min_width_shrink_rate
        << ", max_width_shrink_rate: " << max_width_shrink_rate
        << ", min_height_shrink_rate: " << min_height_shrink_rate
        << ", max_height_shrink_rate: " << max_height_shrink_rate;
    check_failed = true;
  }

  float min_crop_aspect_ratio = conf.attr<float>("min_crop_aspect_ratio");
  float max_crop_aspect_ratio = conf.attr<float>("max_crop_aspect_ratio");
  if (min_crop_aspect_ratio <= 0.0f || min_crop_aspect_ratio >= max_crop_aspect_ratio) {
    err << ", min_crop_aspect_ratio: " << min_crop_aspect_ratio
        << ", max_crop_aspect_ratio: " << max_crop_aspect_ratio;
    check_failed = true;
  }

  int32_t max_num_attempts = conf.attr<int32_t>("max_num_attempts");
  if (max_num_attempts <= 0) {
    err << ", max_num_attempts: " << max_num_attempts;
    check_failed = true;
  }

  if (check_failed) { return oneflow::Error::CheckFailed() << err.str(); }
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* image_desc = ctx->TensorDesc4ArgNameAndIndex("image", 0);
  user_op::TensorDesc* out_image_desc = ctx->TensorDesc4ArgNameAndIndex("out_image", 0);
  CHECK_OR_RETURN(image_desc->data_type() == DataType::kTensorBuffer);
  CHECK_OR_RETURN(image_desc->shape().NumAxes() == 1);
  *out_image_desc->mut_shape() = image_desc->shape();
  *out_image_desc->mut_data_type() = DataType::kTensorBuffer;
  out_image_desc->set_is_dynamic(image_desc->is_dynamic());

  const user_op::TensorDesc* bbox_desc = ctx->TensorDesc4ArgNameAndIndex("bbox", 0);
  user_op::TensorDesc* out_bbox_desc = ctx->TensorDesc4ArgNameAndIndex("out_bbox", 0);
  if (bbox_desc) {
    CHECK_OR_RETURN(bbox_desc->data_type() == DataType::kTensorBuffer);
    CHECK_OR_RETURN(bbox_desc->shape().NumAxes() == 1);
    CHECK_NOTNULL_OR_RETURN(out_bbox_desc);
    *out_bbox_desc->mut_shape() = bbox_desc->shape();
    *out_bbox_desc->mut_data_type() = DataType::kTensorBuffer;
    out_bbox_desc->set_is_dynamic(bbox_desc->is_dynamic());
  }

  const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
  user_op::TensorDesc* out_label_desc = ctx->TensorDesc4ArgNameAndIndex("out_label", 0);
  if (label_desc) {
    CHECK_OR_RETURN(label_desc->data_type() == DataType::kTensorBuffer);
    CHECK_OR_RETURN(label_desc->shape().NumAxes() == 1);
    CHECK_NOTNULL_OR_RETURN(out_label_desc);
    *out_label_desc->mut_shape() = label_desc->shape();
    *out_label_desc->mut_data_type() = DataType::kTensorBuffer;
    out_label_desc->set_is_dynamic(label_desc->is_dynamic());
  }

  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("image", 0)->value(), 0);
  ctx->BatchAxis4ArgNameAndIndex("out_image", 0)->set_value(0);
  auto has_tensor = [ctx](const std::string& bn) -> bool {
    bool ret = false;
    for (auto t : ctx->inputs()) {
      if (bn == t.first) { return true; }
    }
    for (auto t : ctx->outputs()) {
      if (bn == t.first) { return true; }
    }
    return ret;
  };
  if (has_tensor("out_bbox")) { ctx->BatchAxis4ArgNameAndIndex("out_bbox", 0)->set_value(0); }
  if (has_tensor("out_label")) { ctx->BatchAxis4ArgNameAndIndex("out_label", 0)->set_value(0); }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("ssd_random_crop")
    .Input("image")
    .OptionalInput("bbox")
    .OptionalInput("label")
    .Output("out_image")
    .OptionalOutput("out_bbox")
    .OptionalOutput("out_label")
    .Attr<std::vector<float>>("min_iou_overlaps", UserOpAttrType::kAtListFloat,
                              {0.0f, 0.1f, 0.3f, 0.7f, 0.9f, -1.0f})
    .Attr<std::vector<float>>("max_iou_overlaps", UserOpAttrType::kAtListFloat,
                              {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f})
    .Attr<float>("min_width_shrink_rate", UserOpAttrType::kAtFloat, 0.3f)
    .Attr<float>("max_width_shrink_rate", UserOpAttrType::kAtFloat, 1.0f)
    .Attr<float>("min_height_shrink_rate", UserOpAttrType::kAtFloat, 0.3f)
    .Attr<float>("max_height_shrink_rate", UserOpAttrType::kAtFloat, 1.0f)
    .Attr<float>("min_crop_aspect_ratio", UserOpAttrType::kAtFloat, 0.5f)
    .Attr<float>("max_crop_aspect_ratio", UserOpAttrType::kAtFloat, 2.0f)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<bool>("has_seed", UserOpAttrType::kAtBool, false)
    .Attr<int32_t>("max_num_attempts", UserOpAttrType::kAtInt32, 50)
    .SetCheckAttrFn(CheckOpAttr)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetBatchAxisInferFn(InferBatchAxis)
    .SetGetSbpFn(GetSbpSignatures);

}  // namespace oneflow
