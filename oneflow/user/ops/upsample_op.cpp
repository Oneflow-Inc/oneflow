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

namespace {
using namespace oneflow;
template<int32_t N>
typename std::enable_if<(N <= 3), Maybe<void>>::type UpsamplingInferLogicalDesc(
    user_op::InferContext* ctx, const std::string& func_name) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  if (N == 1) {
    CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                    && x_desc.shape().NumAxes() == (N + 2))
        << func_name << " only supports NCH";
    int64_t input_width = x_desc.shape().At(2);
    int64_t output_width = 0;
    const double scale_factor = ctx->Attr<double>("scale_factor");
    std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    if (output_size.size()) {
      output_width = output_size[0];
    } else {
      output_width = static_cast<int64_t>(scale_factor * input_width);
    }
    CHECK_OR_RETURN(input_width > 0 && output_width > 0)
        << func_name
        << ": Input and output sizes should be greater than 0, but got input (W: " << input_width
        << ") output (W: " << output_width << ")";
    y_desc->set_shape(Shape({x_desc.shape().At(0), x_desc.shape().At(1), output_width}));
  } else if (N == 2) {
    CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                    && x_desc.shape().NumAxes() == (N + 2))
        << func_name << " only supports NCHW";
    const double height_scale = ctx->Attr<double>("height_scale");
    const double width_scale = ctx->Attr<double>("width_scale");
    std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    int64_t input_height = x_desc.shape().At(2);
    int64_t input_width = x_desc.shape().At(3);
    int64_t output_height = 0;
    int64_t output_width = 0;
    if (output_size.size()) {
      output_height = output_size[0];
      output_width = output_size[1];
    } else {
      output_height = static_cast<int64_t>(height_scale * input_height);
      output_width = static_cast<int64_t>(width_scale * input_width);
    }
    CHECK_OR_RETURN(input_height > 0 && input_width > 0 && output_height > 0 && output_width > 0)
        << func_name
        << ": Input and output sizes should be greater than 0, but got input (H: " << input_height
        << ", W: " << input_width << ") output (H: " << output_height << ", W: " << output_width
        << ")";
    y_desc->set_shape(
        Shape({x_desc.shape().At(0), x_desc.shape().At(1), output_height, output_width}));
  } else if (N == 3) {
    CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                    && x_desc.shape().NumAxes() == 5)
        << func_name << " only supports NCDHW";
    const double depth_scale = ctx->Attr<double>("depth_scale");
    const double height_scale = ctx->Attr<double>("height_scale");
    const double width_scale = ctx->Attr<double>("width_scale");
    std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    int64_t input_depth = x_desc.shape().At(2);
    int64_t input_height = x_desc.shape().At(3);
    int64_t input_width = x_desc.shape().At(4);
    int64_t output_depth = 0;
    int64_t output_height = 0;
    int64_t output_width = 0;
    if (output_size.size()) {
      output_depth = output_size[0];
      output_height = output_size[1];
      output_width = output_size[2];
    } else {
      output_depth = static_cast<int64_t>(depth_scale * input_depth);
      output_height = static_cast<int64_t>(height_scale * input_height);
      output_width = static_cast<int64_t>(width_scale * input_width);
    }
    CHECK_OR_RETURN(input_depth > 0 && input_height > 0 && input_width > 0 && output_depth > 0
                    && output_height > 0 && output_width > 0)
        << func_name
        << ": Input and output sizes should be greater than 0, but got input (D: " << input_depth
        << ", H: " << input_height << ", W: " << input_width << ") output (D: " << output_depth
        << "H: " << output_height << ", W: " << output_width << ")";
    y_desc->set_shape(Shape(
        {x_desc.shape().At(0), x_desc.shape().At(1), output_depth, output_height, output_width}));
  }
  return Maybe<void>::Ok();
}
}  // namespace

namespace oneflow {

/*static*/ Maybe<void> UpsampleLinear1DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleLinear1DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<1>(ctx, "upsample_linear_1d");
}
/*static*/ Maybe<void> UpsampleLinear1DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleLinear1DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest1DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest1DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<1>(ctx, "upsample_nearest_1d");
}
/*static*/ Maybe<void> UpsampleNearest1DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest1DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest2DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<2>(ctx, "upsample_nearest_2d");
}
/*static*/ Maybe<void> UpsampleNearest2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest2DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleBilinear2DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBilinear2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<2>(ctx, "upsample_bilinear_2d");
}
/*static*/ Maybe<void> UpsampleBilinear2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleBilinear2DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleBicubic2DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBicubic2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<2>(ctx, "upsample_bicubic_2d");
}
/*static*/ Maybe<void> UpsampleBicubic2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleBicubic2DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest3DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest3DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<3>(ctx, "upsample_nearest_3d");
}
/*static*/ Maybe<void> UpsampleNearest3DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest3DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleTrilinear3DOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleTrilinear3DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UpsamplingInferLogicalDesc<3>(ctx, "upsample_trilinear_3d");
}
/*static*/ Maybe<void> UpsampleTrilinear3DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleTrilinear3DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleLinear1DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleLinear1DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 3)
      << "upsample_linear_1d_grad only supports NCH";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleLinear1DGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleLinear1DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest1DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest1DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 3)
      << "upsample_nearest_1d_grad only supports NCH";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest1DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest1DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest2DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest2DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 4)
      << "upsample_nearest_2d_grad only supports NCHW";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest2DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest2DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleBilinear2DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBilinear2DGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 4)
      << "upsample_bilinear_2d_grad only supports NCHW";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBilinear2DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleBilinear2DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleBicubic2DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBicubic2DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 4)
      << "upsample_bicubic_2d_grad only supports NCHW";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleBicubic2DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleBicubic2DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleNearest3DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest3DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 5)
      << "upsample_nearest_3d_grad only supports NCDHW";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleNearest3DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleNearest3DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UpsampleTrilinear3DGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleTrilinear3DGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                  && dy_shape.NumAxes() == 5)
      << "upsample_trilinear_3d_grad only supports NCDHW";
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UpsampleTrilinear3DGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UpsampleTrilinear3DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
