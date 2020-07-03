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
    .Attr("annotation_file", UserOpAttrType::kAtString)
    .Attr("image_dir", UserOpAttrType::kAtString)
    .Attr("batch_size", UserOpAttrType::kAtInt64)
    .Attr<bool>("shuffle_after_epoch", UserOpAttrType::kAtBool, true)
    .Attr<int64_t>("random_seed", UserOpAttrType::kAtInt64, -1)
    .Attr<bool>("group_by_ratio", UserOpAttrType::kAtBool, true)
    .Attr<bool>("remove_images_without_annotations", UserOpAttrType::kAtBool, true)
    .Attr<bool>("stride_partition", UserOpAttrType::kAtBool, false)
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
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      for (const auto& out_arg_pair : ctx->outputs()) {
        ctx->BatchAxis4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second)->set_value(0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
