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
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("COCOReader")
    .Output("image")
    .Output("image_id")
    .Output("image_size")
    .Output("gt_bbox")
    .Output("gt_label")
    .Output("gt_segm")
    .Output("gt_segm_index")
    .Attr<int64_t>("session_id")
    .Attr<std::string>("annotation_file")
    .Attr<std::string>("image_dir")
    .Attr<int64_t>("batch_size")
    .Attr<bool>("shuffle_after_epoch", true)
    .Attr<int64_t>("random_seed", -1)
    .Attr<bool>("group_by_ratio", true)
    .Attr<bool>("remove_images_without_annotations", true)
    .Attr<bool>("stride_partition", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const SbpParallel& sbp = ctx->SbpParallel4ArgNameAndIndex("image", 0);
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("image_id", 0));
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("image_size", 0));
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("gt_bbox", 0));
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("gt_label", 0));
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("gt_segm", 0));
      CHECK_OR_RETURN(sbp == ctx->SbpParallel4ArgNameAndIndex("gt_segm_index", 0));

      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      int64_t device_batch_size = batch_size;
      if (sbp.has_split_parallel() && parallel_num > 1) {
        CHECK_EQ_OR_RETURN(device_batch_size % parallel_num, 0);
        device_batch_size /= parallel_num;
      }

      user_op::TensorDesc* image_desc = ctx->TensorDesc4ArgNameAndIndex("image", 0);
      *image_desc->mut_shape() = Shape({device_batch_size});
      *image_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* image_id_desc = ctx->TensorDesc4ArgNameAndIndex("image_id", 0);
      *image_id_desc->mut_shape() = Shape({device_batch_size});
      *image_id_desc->mut_data_type() = DataType::kInt64;
      user_op::TensorDesc* image_size_desc = ctx->TensorDesc4ArgNameAndIndex("image_size", 0);
      *image_size_desc->mut_shape() = Shape({device_batch_size, 2});
      *image_size_desc->mut_data_type() = DataType::kInt32;
      user_op::TensorDesc* bbox_desc = ctx->TensorDesc4ArgNameAndIndex("gt_bbox", 0);
      *bbox_desc->mut_shape() = Shape({device_batch_size});
      *bbox_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("gt_label", 0);
      *label_desc->mut_shape() = Shape({device_batch_size});
      *label_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* segm_desc = ctx->TensorDesc4ArgNameAndIndex("gt_segm", 0);
      *segm_desc->mut_shape() = Shape({device_batch_size});
      *segm_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* segm_index_desc = ctx->TensorDesc4ArgNameAndIndex("gt_segm_index", 0);
      *segm_index_desc->mut_shape() = Shape({device_batch_size});
      *segm_index_desc->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* image_modifier = GetOutputArgModifierFn("image", 0);
      CHECK(image_modifier != nullptr);
      image_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* image_id_modifier = GetOutputArgModifierFn("image_id", 0);
      CHECK(image_id_modifier != nullptr);
      image_id_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* image_size_modifier = GetOutputArgModifierFn("image_size", 0);
      CHECK(image_size_modifier != nullptr);
      image_size_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* gt_bbox_modifier = GetOutputArgModifierFn("gt_bbox", 0);
      CHECK(gt_bbox_modifier != nullptr);
      gt_bbox_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* gt_label_modifier = GetOutputArgModifierFn("gt_label", 0);
      CHECK(gt_label_modifier != nullptr);
      gt_label_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* gt_segm_modifier = GetOutputArgModifierFn("gt_segm", 0);
      CHECK(gt_segm_modifier != nullptr);
      gt_segm_modifier->set_header_infered_before_compute(false);

      user_op::OutputArgModifier* gt_segm_index_modifier =
          GetOutputArgModifierFn("gt_segm_index", 0);
      CHECK(gt_segm_index_modifier != nullptr);
      gt_segm_index_modifier->set_header_infered_before_compute(false);
    });

}  // namespace oneflow
