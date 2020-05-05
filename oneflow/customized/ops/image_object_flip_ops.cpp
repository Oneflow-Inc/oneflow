#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {}  // namespace

REGISTER_USER_OP("object_bbox_flip")
    .Input("bbox")
    .Input("image_size")
    .Input("flip_code")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* bbox_desc = ctx->TensorDesc4ArgNameAndIndex("bbox", 0);
      CHECK_EQ_OR_RETURN(bbox_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(bbox_desc->shape().NumAxes(), 1);
      const int N = bbox_desc->shape().elem_cnt();

      const user_op::TensorDesc* image_size_desc = ctx->TensorDesc4ArgNameAndIndex("image_size", 0);
      CHECK_EQ_OR_RETURN(image_size_desc->data_type(), DataType::kInt32);
      CHECK_EQ_OR_RETURN(image_size_desc->shape().elem_cnt(), N * 2);

      const user_op::TensorDesc* flip_code_desc = ctx->TensorDesc4ArgNameAndIndex("flip_code", 0);
      CHECK_EQ_OR_RETURN(flip_code_desc->data_type(), DataType::kInt8);
      CHECK_EQ_OR_RETURN(flip_code_desc->shape().elem_cnt(), N);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *bbox_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("bbox", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("object_segmentation_polygon_flip")
    .Input("polygon")
    .Input("image_size")
    .Input("flip_code")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* polygon_desc = ctx->TensorDesc4ArgNameAndIndex("polygon", 0);
      CHECK_EQ_OR_RETURN(polygon_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(polygon_desc->shape().NumAxes(), 1);
      const int N = polygon_desc->shape().elem_cnt();

      const user_op::TensorDesc* image_size_desc = ctx->TensorDesc4ArgNameAndIndex("image_size", 0);
      CHECK_EQ_OR_RETURN(image_size_desc->data_type(), DataType::kInt32);
      CHECK_EQ_OR_RETURN(image_size_desc->shape().elem_cnt(), N * 2);

      const user_op::TensorDesc* flip_code_desc = ctx->TensorDesc4ArgNameAndIndex("flip_code", 0);
      CHECK_EQ_OR_RETURN(flip_code_desc->data_type(), DataType::kInt8);
      CHECK_EQ_OR_RETURN(flip_code_desc->shape().elem_cnt(), N);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *polygon_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("polygon", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
