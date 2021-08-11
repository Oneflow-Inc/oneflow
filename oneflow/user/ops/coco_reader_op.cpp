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

REGISTER_NO_GRAD_CPU_ONLY_USER_OP("COCOReader")
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
    .Attr<std::vector<std::string>>("parallel_distribution")
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const cfg::SbpParallel& sbp = ctx->SbpParallel4ArgNameAndIndex("image", 0);
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

      user_op::TensorDesc* image_desc = ctx->OutputTensorDesc("image", 0);
      *image_desc->mut_shape() = Shape({device_batch_size});
      user_op::TensorDesc* image_id_desc = ctx->OutputTensorDesc("image_id", 0);
      *image_id_desc->mut_shape() = Shape({device_batch_size});
      user_op::TensorDesc* image_size_desc = ctx->OutputTensorDesc("image_size", 0);
      *image_size_desc->mut_shape() = Shape({device_batch_size, 2});
      user_op::TensorDesc* bbox_desc = ctx->OutputTensorDesc("gt_bbox", 0);
      *bbox_desc->mut_shape() = Shape({device_batch_size});
      user_op::TensorDesc* label_desc = ctx->OutputTensorDesc("gt_label", 0);
      *label_desc->mut_shape() = Shape({device_batch_size});
      user_op::TensorDesc* segm_desc = ctx->OutputTensorDesc("gt_segm", 0);
      *segm_desc->mut_shape() = Shape({device_batch_size});
      user_op::TensorDesc* segm_index_desc = ctx->OutputTensorDesc("gt_segm_index", 0);
      *segm_index_desc->mut_shape() = Shape({device_batch_size});
      return Maybe<void>::Ok();
    })
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      user_op::TensorDesc* image_desc = ctx->OutputTensorDesc("image", 0);
      *image_desc->mut_shape() = Shape({batch_size});
      user_op::TensorDesc* image_id_desc = ctx->OutputTensorDesc("image_id", 0);
      *image_id_desc->mut_shape() = Shape({batch_size});
      user_op::TensorDesc* image_size_desc = ctx->OutputTensorDesc("image_size", 0);
      *image_size_desc->mut_shape() = Shape({batch_size, 2});
      user_op::TensorDesc* bbox_desc = ctx->OutputTensorDesc("gt_bbox", 0);
      *bbox_desc->mut_shape() = Shape({batch_size});
      user_op::TensorDesc* label_desc = ctx->OutputTensorDesc("gt_label", 0);
      *label_desc->mut_shape() = Shape({batch_size});
      user_op::TensorDesc* segm_desc = ctx->OutputTensorDesc("gt_segm", 0);
      *segm_desc->mut_shape() = Shape({batch_size});
      user_op::TensorDesc* segm_index_desc = ctx->OutputTensorDesc("gt_segm_index", 0);
      *segm_index_desc->mut_shape() = Shape({batch_size});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const auto& dist_conf =
              ctx->user_op_conf().attr<std::vector<std::string>>("parallel_distribution");
          const Shape& hierarchy = ctx->parallel_hierarchy();

          // the inputs may be produced by tick and others, whose sbp should be broadcast parallel
          // dist
          for (const auto& arg_pair : ctx->inputs()) {
            cfg::ParallelDistribution* input_dist =
                ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
            FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
              input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
            }
          }

          for (const auto& arg_pair : ctx->outputs()) {
            cfg::ParallelDistribution* output_dist =
                ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
            if (dist_conf.size() == 0) {
              // the default parallel dist of outputs of dataset op should be split(0)
              FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
                output_dist->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
              }
            } else {
              CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
              for (const std::string& sbp_str : dist_conf) {
                cfg::SbpParallel sbp_parallel;
                CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
                CHECK_OR_RETURN(
                    (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0)
                    || sbp_parallel.has_broadcast_parallel());
                *output_dist->add_sbp_parallel() = sbp_parallel;
              }
            }
          }
          return Maybe<void>::Ok();
        })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
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

      user_op::OutputArgModifier* gt_segm_index_modifier =
          GetOutputArgModifierFn("gt_segm_index", 0);
      CHECK_OR_RETURN(gt_segm_index_modifier != nullptr);
      gt_segm_index_modifier->set_header_infered_before_compute(false);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* image_desc = ctx->OutputTensorDesc("image", 0);
      *image_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* image_id_desc = ctx->OutputTensorDesc("image_id", 0);
      *image_id_desc->mut_data_type() = DataType::kInt64;
      user_op::TensorDesc* image_size_desc = ctx->OutputTensorDesc("image_size", 0);
      *image_size_desc->mut_data_type() = DataType::kInt32;
      user_op::TensorDesc* bbox_desc = ctx->OutputTensorDesc("gt_bbox", 0);
      *bbox_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* label_desc = ctx->OutputTensorDesc("gt_label", 0);
      *label_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* segm_desc = ctx->OutputTensorDesc("gt_segm", 0);
      *segm_desc->mut_data_type() = DataType::kTensorBuffer;
      user_op::TensorDesc* segm_index_desc = ctx->OutputTensorDesc("gt_segm_index", 0);
      *segm_index_desc->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
