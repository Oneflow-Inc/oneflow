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

/* static */ Maybe<void> DeformConv2dOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);

  const int32_t kW = weight_shape.at(3);
  const int32_t kH = weight_shape.at(2);
  const int32_t dW = ctx->Attr<int32_t>("stride_w");
  const int32_t dH = ctx->Attr<int32_t>("stride_h");
  const int32_t padW = ctx->Attr<int32_t>("pad_w");
  const int32_t padH = ctx->Attr<int32_t>("pad_h");
  const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
  const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4);
  CHECK_EQ_OR_RETURN(weight_shape.At(2), kH);
  CHECK_EQ_OR_RETURN(weight_shape.At(3), kW);
  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  *ctx->MutOutputShape("output", 0) =
      Shape({input_shape.At(0), weight_shape.At(0), outputHeight, outputWidth});

  *ctx->MutOutputIsDynamic("output", 0) = ctx->InputIsDynamic("input", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dInputGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const Shape& offset_shape = ctx->InputShape("offset", 0);
  const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);
  const int32_t kW = weight_shape.at(3);
  const int32_t kH = weight_shape.at(2);
  const int32_t dW = ctx->Attr<int32_t>("stride_w");
  const int32_t dH = ctx->Attr<int32_t>("stride_h");
  const int32_t padW = ctx->Attr<int32_t>("pad_w");
  const int32_t padH = ctx->Attr<int32_t>("pad_h");
  const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
  const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
  const bool use_mask = ctx->Attr<bool>("use_mask");
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4)
      << Error::RuntimeError() << "The dimension of tensor weight must be 4,but got "
      << weight_shape.NumAxes();

  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(1), weight_shape.At(0));
  *ctx->MutOutputShape("input_grad", 0) = ctx->InputShape("input", 0);
  *ctx->MutOutputShape("offset_grad", 0) = ctx->InputShape("offset", 0);
  *ctx->MutOutputIsDynamic("input_grad", 0) = ctx->InputIsDynamic("input", 0);
  *ctx->MutOutputIsDynamic("offset_grad", 0) = false;
  if (use_mask) {
    *ctx->MutOutputShape("mask_grad", 0) =
        Shape({offset_shape.At(0), offset_shape.At(1) / 2, offset_shape.At(2), offset_shape.At(3)});
    *ctx->MutOutputIsDynamic("mask_grad", 0) = false;
  }
  return Maybe<void>::Ok();
}

Maybe<void> DeformConv2dParamGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const int32_t kW = weight_shape.at(3);
  const int32_t kH = weight_shape.at(2);
  const int32_t dW = ctx->Attr<int32_t>("stride_w");
  const int32_t dH = ctx->Attr<int32_t>("stride_h");
  const int32_t padW = ctx->Attr<int32_t>("pad_w");
  const int32_t padH = ctx->Attr<int32_t>("pad_h");
  const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
  const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);
  *ctx->MutOutputShape("weight_grad", 0) = ctx->InputShape("weight", 0);
  *ctx->MutOutputIsDynamic("weight_grad", 0) = false;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DeformConv2dInputGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DeformConv2dParamGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DeformConv2dOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dInputGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("input_grad", 0) = ctx->InputDType("input", 0);
  *ctx->MutOutputDType("offset_grad", 0) = ctx->InputDType("offset", 0);
  const bool use_mask = ctx->Attr<bool>("use_mask");

  if (use_mask) { *ctx->MutOutputDType("mask_grad", 0) = ctx->InputDType("mask", 0); }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dParamGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("weight_grad", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dOp::CheckAttr(const user_op::UserOpDefWrapper& op_def,
                                                   const user_op::UserOpConfWrapper& op_conf) {
  const int32_t dW = op_conf.attr<int32_t>("stride_w");
  const int32_t dH = op_conf.attr<int32_t>("stride_h");
  const int32_t dilationH = op_conf.attr<int32_t>("dilation_h");
  const int32_t dilationW = op_conf.attr<int32_t>("dilation_w");

  CHECK_OR_RETURN(dW > 0 && dH > 0)
      << Error::RuntimeError() << "The stride must be greater than 0,but got " << dW << " and "
      << dH;

  CHECK_OR_RETURN(dilationW > 0 && dilationH > 0)
      << Error::RuntimeError() << "The dilation must be greater than 0,but got " << dilationH
      << " and " << dilationW;

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("bias", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("output", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dInputGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("output_grad", 0), 0)
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("input_grad", 0), 0)
      .Split(user_op::OpArg("offset_grad", 0), 0)
      .Split(user_op::OpArg("mask_grad", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dParamGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("output_grad", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .PartialSum(user_op::OpArg("weight_grad", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow