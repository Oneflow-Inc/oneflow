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

Maybe<void> Deformconv2dOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const int32_t kW = ctx->Attr<int32_t>("kW");
  const int32_t kH = ctx->Attr<int32_t>("kH");
  const int32_t dW = ctx->Attr<int32_t>("dW");
  const int32_t dH = ctx->Attr<int32_t>("dH");
  const int32_t padW = ctx->Attr<int32_t>("padW");
  const int32_t padH = ctx->Attr<int32_t>("padH");
  const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
  const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4);
  CHECK_EQ_OR_RETURN(weight_shape.At(2), kH);
  CHECK_EQ_OR_RETURN(weight_shape.At(3), kW);
  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  *ctx->OutputShape("output", 0) =
      Shape({input_shape.At(0), input_shape.At(1), outputHeight, outputWidth});
  *ctx->OutputIsDynamic("output", 0) = ctx->InputIsDynamic("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> DeformConv2dInputGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);
  const int32_t kW = ctx->Attr<int32_t>("kW");
  const int32_t kH = ctx->Attr<int32_t>("kH");
  const int32_t dW = ctx->Attr<int32_t>("dW");
  const int32_t dH = ctx->Attr<int32_t>("dH");
  const int32_t padW = ctx->Attr<int32_t>("padW");
  const int32_t padH = ctx->Attr<int32_t>("padH");
  const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
  const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 4);
  CHECK_EQ_OR_RETURN(weight_shape.At(2), kH);
  CHECK_EQ_OR_RETURN(weight_shape.At(3), kW);
  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(1), weight_shape.At(0));
  *ctx->OutputShape("input_grad", 0) = ctx->InputShape("input", 0);
  *ctx->OutputShape("offset_grad", 0) = ctx->InputShape("offset", 0);
  *ctx->OutputIsDynamic("input_grad", 0) = ctx->InputIsDynamic("input", 0);
  *ctx->OutputIsDynamic("offset_grad", 0) = false;
  return Maybe<void>::Ok();
}

Maybe<void> Deformconv2dParamGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);
  const int32_t kW = ctx->Attr<int32_t>("kW");
  const int32_t kH = ctx->Attr<int32_t>("kH");
  const int32_t dW = ctx->Attr<int32_t>("dW");
  const int32_t dH = ctx->Attr<int32_t>("dH");
  const int32_t padW = ctx->Attr<int32_t>("padW");
  const int32_t padH = ctx->Attr<int32_t>("padH");
  const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
  const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
  const int32_t group = ctx->Attr<int32_t>("group");
  int64_t outputWidth = (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  CHECK_EQ_OR_RETURN(output_grad_shape.At(2), outputHeight);
  CHECK_EQ_OR_RETURN(output_grad_shape.At(3), outputWidth);
  *ctx->OutputShape("weight_grad", 0) =
      Shape({output_grad_shape.At(1), input_shape.At(1) / group, kH, kW});
  *ctx->OutputIsDynamic("weight_grad", 0) = false;
  return Maybe<void>::Ok();
}

Maybe<void> Deformconv2dOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx){return InferLogicalTensorDesc(
    ctx)} Maybe<void> DeformConv2dInputGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx){
    return InferLogicalTensorDesc(
        ctx)} Maybe<void> Deformconv2dParamGradOp::InferPhysicalTensorDesc(user_op::InferContext*
                                                                               ctx){
    return InferLogicalTensorDesc(ctx)}

Maybe<void> Deformconv2dOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}
Maybe<void> DeformConv2dInputGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("input_grad", 0) = ctx->InputDType("input", 0);
  *ctx->OutputDType("offset_grad", 0) = ctx->InputDType("offset", 0);
  return Maybe<void>::Ok();
}
Maybe<void> Deformconv2dParamGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("weight_grad", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> Deformconv2dOp::CheckAttr(onst user_op::UserOpDefWrapper& op_def,
                                      const user_op::UserOpConfWrapper& op_conf) {
  bool is_checked = true;
  std::stringstream err;
  err << "Illegal value for " << op_conf.op_type_name() << "op " << op_conf.op_name() << ": ";

  const int32_t kW = op_conf.attr<int32_t>("kW");
  const int32_t kH = op_conf.attr<int32_t>("kH");
  const int32_t dW = op_conf.attr<int32_t>("dW");
  const int32_t dH = op_conf.attr<int32_t>("dH");
  const int32_t dilationH = op_conf.attr<int32_t>("dilationH");
  const int32_t dilationW = op_conf.attr<int32_t>("dilationW");
  if (!(kW > 0 && kH > 0)) {
    err << " kernel_size: "
        << "(" << kH << ", " << kW << ")";
    is_checked = false;
  }
  if (!(dW > 0 && dH > 0)) {
    err << "stride: "
        << "(" << dH << ", " << dW << ")";
    is_checked = false;
  }
  if (!(dilationW > 0 && dilationH > 0)) {
    err << "dilation: "
        << "(" << dilationH << ", " << dilationW << ")";
    is_checked = false;
  }

  if (is_checked) {
    return Maybe<void>::Ok();
  } else {
    return oneflow::Error::CheckFailedError() << err.str();
  }
}
Maybe<void> Deformconv2dOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("output", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> DeformConv2dInputGradSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("output_grad", 0), 0)
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("input_grad", 0), 0)
      .Split(user_op::OpArg("offset_grad", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> DeformConv2dParamGradSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("output_grad", 0), 0)
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("offset", 0), 0)
      .PartialSum(user_op::OpArg("weight_grad", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4DeformConv(const user_op::UserOpWrapper& op,
                                              user_op::AddOpFn AddOp) {
  const int32_t kW = op.attr<int32_t>("kW");
  const int32_t kH = op.attr<int32_t>("kH");
  const int32_t dW = op.attr<int32_t>("dW");
  const int32_t dH = op.attr<int32_t>("dH");
  const int32_t padW = op.attr<int32_t>("padW");
  const int32_t padH = op.attr<int32_t>("padH");
  const int32_t dilationW = op.attr<int32_t>("dilationW");
  const int32_t dilationH = op.attr<int32_t>("dilationH");
  const int32_t group = op.attr<int32_t>("group");
  const int32_t deformable_group = op.attr<int32_t>("deformable_group");

  if (op.NeedGenGradTensor4OpInput("input", 0) || op.NeedGenGradTensor4OpInput("offset", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_input_grad");
    user_op::UserOpConfWrapper input_grad_op =
        builder.Op("deform_conv2d_input_grad")
            .Input("output_grad", op.GetGradTensorWithOpOutput("output", 0))
            .Input("input", op.input("input", 0))
            .Input("weight", op.input("weight", 0))
            .Input("offset", op.input("offset", 0))
            .Output("input_grad")
            .Output("offset_grad")
            .Attr<int32_t>("kW", kW)
            .Attr<int32_t>("kH", kH)
            .Attr<int32_t>("dW", dW)
            .Attr<int32_t>("dH", dH)
            .Attr<int32_t>("padW", padW)
            .Attr<int32_t>("padH", padH)
            .Attr<int32_t>("dilationW", dilationW)
            .Attr<int32_t>("dilationH", dilationH)
            .Attr<int32_t>("group", group)
            .Attr<int32_t>("deformable_group", deformable_group)
            .Build();
    if (op.NeedGenGradTensor4OpInput("input", 0)) {
      op.BindGradTensorWithOpInput(input_grad_op.output("input_grad", 0), "input", 0);
    }
    if (op.NeedGenGradTensor4OpInput("offset", 0)) {
      op.BindGradTensorWithOpInput(input_grad_op.output("offset_grad", 0), "offset", 0);
    }
    AddOp(input_grad_op);
  }

  if (op.NeedGenGradTensor4OpInput("weight", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_param_grad");
    user_op::UserOpConfWrapper param_grad_op =
        builder.Op("deform_conv2d_param_grad")
            .Input("output_grad", op.GetGradTensorWithOpOutput("output", 0))
            .Input("input", op.input("input", 0))
            .Input("offset", op.input("offset", 0))
            .Output("weight_grad")
            .Attr<int32_t>("kW", kW)
            .Attr<int32_t>("kH", kH)
            .Attr<int32_t>("dW", dW)
            .Attr<int32_t>("dH", dH)
            .Attr<int32_t>("padW", padW)
            .Attr<int32_t>("padH", padH)
            .Attr<int32_t>("dilationW", dilationW)
            .Attr<int32_t>("dilationH", dilationH)
            .Attr<int32_t>("group", group)
            .Attr<int32_t>("deformable_group", deformable_group)
            .Build();
    op.BindGradTensorWithOpInput(param_grad_op.output("weight_grad", 0), "weight", 0);
    AddOp(param_grad_op);
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP_GRAD("deform_conv2d").SetGenBackwardOpConfFn(GenerateBackwardOpConf4DeformConv);

}  // namespace oneflow