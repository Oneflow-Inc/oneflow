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

namespace {

Maybe<void> ImageObjectGetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> ImageFlipOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_desc.shape().NumAxes(), 1);
  const int N = in_desc.shape().elem_cnt();

  const user_op::TensorDesc& flip_code_desc = ctx->InputTensorDesc("flip_code", 0);
  CHECK_EQ_OR_RETURN(flip_code_desc.shape().elem_cnt(), N);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImageFlipOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ImageFlipOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ImageFlipOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_desc.data_type(), DataType::kTensorBuffer);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ObjectBboxFlipOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& bbox_desc = ctx->InputTensorDesc("bbox", 0);
  CHECK_EQ_OR_RETURN(bbox_desc.shape().NumAxes(), 1);
  const int N = bbox_desc.shape().elem_cnt();

  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.shape().elem_cnt(), N * 2);

  const user_op::TensorDesc& flip_code_desc = ctx->InputTensorDesc("flip_code", 0);
  CHECK_EQ_OR_RETURN(flip_code_desc.shape().elem_cnt(), N);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("bbox", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("bbox", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ObjectBboxFlipOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ObjectBboxFlipOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ObjectBboxFlipOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& bbox_desc = ctx->InputTensorDesc("bbox", 0);
  CHECK_EQ_OR_RETURN(bbox_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.data_type(), DataType::kInt32);
  const user_op::TensorDesc& flip_code_desc = ctx->InputTensorDesc("flip_code", 0);
  CHECK_EQ_OR_RETURN(flip_code_desc.data_type(), DataType::kInt8);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("bbox", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ObjectBboxScaleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& bbox_desc = ctx->InputTensorDesc("bbox", 0);
  CHECK_EQ_OR_RETURN(bbox_desc.shape().NumAxes(), 1);
  const int N = bbox_desc.shape().elem_cnt();

  const user_op::TensorDesc& scale_desc = ctx->InputTensorDesc("scale", 0);
  CHECK_EQ_OR_RETURN(scale_desc.shape().elem_cnt(), N * 2);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("bbox", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("bbox", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ObjectBboxScaleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ObjectBboxScaleOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ObjectBboxScaleOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& bbox_desc = ctx->InputTensorDesc("bbox", 0);
  CHECK_EQ_OR_RETURN(bbox_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& scale_desc = ctx->InputTensorDesc("scale", 0);
  CHECK_EQ_OR_RETURN(scale_desc.data_type(), DataType::kFloat);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("bbox", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ObjectSegmentationPolygonFlipOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.shape().NumAxes(), 1);
  const int N = poly_desc.shape().elem_cnt();

  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.shape().elem_cnt(), N * 2);

  const user_op::TensorDesc& flip_code_desc = ctx->InputTensorDesc("flip_code", 0);
  CHECK_EQ_OR_RETURN(flip_code_desc.shape().elem_cnt(), N);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("poly", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("poly", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ObjectSegmentationPolygonFlipOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonFlipOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonFlipOp::InferDataType(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.data_type(), DataType::kInt32);
  const user_op::TensorDesc& flip_code_desc = ctx->InputTensorDesc("flip_code", 0);
  CHECK_EQ_OR_RETURN(flip_code_desc.data_type(), DataType::kInt8);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("poly", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ObjectSegmentationPolygonScaleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.shape().NumAxes(), 1);
  const int N = poly_desc.shape().elem_cnt();

  const user_op::TensorDesc& scale_desc = ctx->InputTensorDesc("scale", 0);
  CHECK_EQ_OR_RETURN(scale_desc.shape().elem_cnt(), N * 2);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("poly", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("poly", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ObjectSegmentationPolygonScaleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonScaleOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonScaleOp::InferDataType(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& scale_desc = ctx->InputTensorDesc("scale", 0);
  CHECK_EQ_OR_RETURN(scale_desc.data_type(), DataType::kFloat);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("poly", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ImageNormalizeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_desc.shape().NumAxes(), 1);
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImageNormalizeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ImageNormalizeOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ImageNormalizeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_desc.data_type(), DataType::kTensorBuffer);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ObjectSegmentationPolygonToMaskOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.shape().NumAxes(), 1);
  const int N = poly_desc.shape().elem_cnt();

  const user_op::TensorDesc& poly_index_desc = ctx->InputTensorDesc("poly_index", 0);
  CHECK_EQ_OR_RETURN(poly_index_desc.shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(poly_index_desc.shape().elem_cnt(), N);

  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.shape().elem_cnt(), N * 2);

  *ctx->MutOutputShape("out", 0) = ctx->InputShape("poly", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("poly", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ObjectSegmentationPolygonToMaskOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonToMaskOp::GetSbp(user_op::SbpContext* ctx) {
  return ImageObjectGetSbp(ctx);
}

/* static */ Maybe<void> ObjectSegmentationPolygonToMaskOp::InferDataType(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& poly_desc = ctx->InputTensorDesc("poly", 0);
  CHECK_EQ_OR_RETURN(poly_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& poly_index_desc = ctx->InputTensorDesc("poly_index", 0);
  CHECK_EQ_OR_RETURN(poly_index_desc.data_type(), DataType::kTensorBuffer);
  const user_op::TensorDesc& image_size_desc = ctx->InputTensorDesc("image_size", 0);
  CHECK_EQ_OR_RETURN(image_size_desc.data_type(), DataType::kInt32);
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("poly", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
