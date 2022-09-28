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
  const Shape& offset_shape = ctx->InputShape("offset", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  const int32_t kW = weight_shape.at(3);
  const int32_t kH = weight_shape.at(2);
  const int32_t dW = ctx->Attr<int32_t>("stride_w");
  const int32_t dH = ctx->Attr<int32_t>("stride_h");
  const int32_t padW = ctx->Attr<int32_t>("pad_w");
  const int32_t padH = ctx->Attr<int32_t>("pad_h");
  const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
  const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
  const int32_t deformable_group = ctx->Attr<int32_t>("offset_groups");
  const bool use_mask = ctx->Attr<bool>("use_mask");
  bool has_bias = ctx->has_input("bias", 0);
  if (has_bias) {
    const Shape& bias_shape = ctx->InputShape("bias", 0);
    std::cout << "bias_shape:" << bias_shape.ToString() << std::endl;
    CHECK_EQ_OR_RETURN(bias_shape.At(0), weight_shape.At(0));
  }
  CHECK_OR_RETURN(dW > 0 && dH > 0)
      << Error::RuntimeError() << "The stride must be greater than 0,but got " << dW << " and "
      << dH;
  CHECK_OR_RETURN(kW > 0 && kH > 0)
      << Error::RuntimeError() << "The weight must be greater than 0,but got " << kW << " and "
      << kH;

  CHECK_OR_RETURN(padW >= 0 && padH >= 0)
      << Error::RuntimeError() << "The pad must be greater than or equal to 0,but got " << padW
      << " and " << padH;
  CHECK_OR_RETURN(dilationW > 0 && dilationH > 0)
      << Error::RuntimeError() << "The dilation must be greater than 0,but got " << dilationH
      << " and " << dilationW;

  CHECK_EQ_OR_RETURN(input_shape.NumAxes(), 4);                   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4);                  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(offset_shape.NumAxes(), 4);                  // NOLINT(maybe-need-error-msg)
  if (use_mask) { CHECK_EQ_OR_RETURN(mask_shape.NumAxes(), 4); }  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(weight_shape.At(2), kH);                     // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(weight_shape.At(3), kW);                     // NOLINT(maybe-need-error-msg)

  CHECK_EQ_OR_RETURN(offset_shape.At(1), deformable_group * 2 * kW * kH)
      << Error::RuntimeError() << "offset.shape[1] is not valid: got: " << offset_shape.At(1)
      << " ,expected: " << deformable_group * 2 * kW * kH;

  if (use_mask) {
    CHECK_EQ_OR_RETURN(mask_shape.At(1), deformable_group * kW * kH)
        << Error::RuntimeError() << "mask.shape[1] is not valid: got: " << mask_shape.At(1)
        << " expected: " << deformable_group * kW * kH;
  }
  CHECK_EQ_OR_RETURN(offset_shape.At(0), input_shape.At(0))
      << Error::RuntimeError() << "invalid batch size of offset:got: " << offset_shape.At(0)
      << " ,expected: " << input_shape.At(0);

  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_OR_RETURN(outputWidth > 0 && outputHeight > 0)
      << Error::RuntimeError() << "Calculated output size too small - out_h: " << outputHeight
      << " ,out_w: " << outputWidth;
  CHECK_OR_RETURN(offset_shape.At(2) == outputHeight && offset_shape.At(3) == outputWidth)
      << Error::RuntimeError() << "invalid offset output dims: got ( " << offset_shape.At(2) << ", "
      << offset_shape.At(3) << ")"
      << ",expected: "
      << "(" << outputHeight << ", " << outputWidth << ")";

  if (use_mask) {
    CHECK_OR_RETURN(mask_shape.At(2) == outputHeight && mask_shape.At(3) == outputWidth)
        << Error::RuntimeError() << "invalid mask output dims: got ( " << mask_shape.At(2) << ", "
        << mask_shape.At(3) << ")"
        << ",expected: "
        << "(" << outputHeight << ", " << outputWidth << ")";
  }
  ctx->SetOutputShape("output", 0,
                      Shape({input_shape.At(0), weight_shape.At(0), outputHeight, outputWidth}));
  ctx->SetOutputIsDynamic("output", 0, ctx->InputIsDynamic("input", 0));

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
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4);

  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);         // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(output_grad_shape.At(1), weight_shape.At(0));  // NOLINT(maybe-need-error-msg)
  ctx->SetOutputShape("input_grad", 0, ctx->InputShape("input", 0));
  ctx->SetOutputShape("offset_grad", 0, ctx->InputShape("offset", 0));
  ctx->SetOutputIsDynamic("input_grad", 0, ctx->InputIsDynamic("input", 0));
  ctx->SetOutputIsDynamic("offset_grad", 0, false);

  if (use_mask) {
    ctx->SetOutputShape("mask_grad", 0,
                        Shape({offset_shape.At(0), offset_shape.At(1) / 2, offset_shape.At(2),
                               offset_shape.At(3)}));
    ctx->SetOutputIsDynamic("mask_grad", 0, false);
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
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);   // NOLINT(maybe-need-error-msg)
  ctx->SetOutputShape("weight_grad", 0, ctx->InputShape("weight", 0));
  ctx->SetOutputIsDynamic("weight_grad", 0, false);
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
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dInputGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("input_grad", 0, ctx->InputDType("input", 0));
  ctx->SetOutputDType("offset_grad", 0, ctx->InputDType("offset", 0));
  const bool use_mask = ctx->Attr<bool>("use_mask");
  if (use_mask) { ctx->SetOutputDType("mask_grad", 0, ctx->InputDType("mask", 0)); }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dParamGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("weight_grad", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DeformConv2dOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
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
