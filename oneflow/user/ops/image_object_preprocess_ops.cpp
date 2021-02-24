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

Maybe<void> GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_CPU_ONLY_USER_OP("image_flip")
    .Input("in")
    .Input("flip_code")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_EQ_OR_RETURN(in_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(in_desc->shape().NumAxes(), 1);
      const int N = in_desc->shape().elem_cnt();

      const user_op::TensorDesc* flip_code_desc = ctx->TensorDesc4ArgNameAndIndex("flip_code", 0);
      CHECK_EQ_OR_RETURN(flip_code_desc->data_type(), DataType::kInt8);
      CHECK_EQ_OR_RETURN(flip_code_desc->shape().elem_cnt(), N);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *in_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("object_bbox_flip")
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
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("object_bbox_scale")
    .Input("bbox")
    .Input("scale")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* bbox_desc = ctx->TensorDesc4ArgNameAndIndex("bbox", 0);
      CHECK_EQ_OR_RETURN(bbox_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(bbox_desc->shape().NumAxes(), 1);
      const int N = bbox_desc->shape().elem_cnt();

      const user_op::TensorDesc* scale_desc = ctx->TensorDesc4ArgNameAndIndex("scale", 0);
      CHECK_EQ_OR_RETURN(scale_desc->data_type(), DataType::kFloat);
      CHECK_EQ_OR_RETURN(scale_desc->shape().elem_cnt(), N * 2);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *bbox_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("object_segmentation_polygon_flip")
    .Input("poly")
    .Input("image_size")
    .Input("flip_code")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* poly_desc = ctx->TensorDesc4ArgNameAndIndex("poly", 0);
      CHECK_EQ_OR_RETURN(poly_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(poly_desc->shape().NumAxes(), 1);
      const int N = poly_desc->shape().elem_cnt();

      const user_op::TensorDesc* image_size_desc = ctx->TensorDesc4ArgNameAndIndex("image_size", 0);
      CHECK_EQ_OR_RETURN(image_size_desc->data_type(), DataType::kInt32);
      CHECK_EQ_OR_RETURN(image_size_desc->shape().elem_cnt(), N * 2);

      const user_op::TensorDesc* flip_code_desc = ctx->TensorDesc4ArgNameAndIndex("flip_code", 0);
      CHECK_EQ_OR_RETURN(flip_code_desc->data_type(), DataType::kInt8);
      CHECK_EQ_OR_RETURN(flip_code_desc->shape().elem_cnt(), N);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *poly_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("object_segmentation_polygon_scale")
    .Input("poly")
    .Input("scale")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* poly_desc = ctx->TensorDesc4ArgNameAndIndex("poly", 0);
      CHECK_EQ_OR_RETURN(poly_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(poly_desc->shape().NumAxes(), 1);
      const int N = poly_desc->shape().elem_cnt();

      const user_op::TensorDesc* scale_desc = ctx->TensorDesc4ArgNameAndIndex("scale", 0);
      CHECK_EQ_OR_RETURN(scale_desc->data_type(), DataType::kFloat);
      CHECK_EQ_OR_RETURN(scale_desc->shape().elem_cnt(), N * 2);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *poly_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("image_normalize")
    .Input("in")
    .Attr<std::vector<float>>("std")
    .Attr<std::vector<float>>("mean")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_EQ_OR_RETURN(in_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(in_desc->shape().NumAxes(), 1);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *in_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

REGISTER_CPU_ONLY_USER_OP("object_segmentation_polygon_to_mask")
    .Input("poly")
    .Input("poly_index")
    .Input("image_size")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* poly_desc = ctx->TensorDesc4ArgNameAndIndex("poly", 0);
      CHECK_EQ_OR_RETURN(poly_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(poly_desc->shape().NumAxes(), 1);
      const int N = poly_desc->shape().elem_cnt();

      const user_op::TensorDesc* poly_index_desc = ctx->TensorDesc4ArgNameAndIndex("poly_index", 0);
      CHECK_EQ_OR_RETURN(poly_index_desc->data_type(), DataType::kTensorBuffer);
      CHECK_EQ_OR_RETURN(poly_index_desc->shape().NumAxes(), 1);
      CHECK_EQ_OR_RETURN(poly_index_desc->shape().elem_cnt(), N);

      const user_op::TensorDesc* image_size_desc = ctx->TensorDesc4ArgNameAndIndex("image_size", 0);
      CHECK_EQ_OR_RETURN(image_size_desc->data_type(), DataType::kInt32);
      CHECK_EQ_OR_RETURN(image_size_desc->shape().elem_cnt(), N * 2);

      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *poly_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbp);

}  // namespace oneflow
