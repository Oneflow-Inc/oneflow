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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> OfrecordRawDecoderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1 && in_tensor.shape().At(0) >= 1);
  Shape conf_shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec(1 + conf_shape.NumAxes());
  dim_vec[0] = in_tensor.shape().At(0);
  for (int i = 1; i < dim_vec.size(); ++i) { dim_vec[i] = conf_shape.At(i - 1); }
  out_tensor->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OfrecordRawDecoderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OfrecordRawDecoderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Split(user_op::OpArg("out", 0), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordRawDecoderOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordRawDecoderOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.data_type() == DataType::kOFRecord);
  out_tensor->set_data_type(ctx->Attr<DataType>("data_type"));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordBytesDecoderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_is_dynamic(in.is_dynamic());
  out->set_shape(in.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OfrecordBytesDecoderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OfrecordBytesDecoderOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}

/* static */ Maybe<void> OfrecordBytesDecoderOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordBytesDecoderOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in.data_type() == DataType::kOFRecord);
  out->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1 && in_tensor.shape().At(0) >= 1);
  out_tensor->set_shape(in_tensor.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OfrecordImageDecoderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OfrecordImageDecoderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Split(user_op::OpArg("out", 0), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.data_type() == DataType::kOFRecord);
  out_tensor->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderRandomCropOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1 && in_tensor.shape().At(0) >= 1);
  out_tensor->set_shape(in_tensor.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OfrecordImageDecoderRandomCropOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OfrecordImageDecoderRandomCropOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Split(user_op::OpArg("out", 0), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderRandomCropOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OfrecordImageDecoderRandomCropOp::InferDataType(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.data_type() == DataType::kOFRecord);
  out_tensor->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
