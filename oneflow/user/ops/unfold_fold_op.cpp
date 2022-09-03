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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> UnfoldTensorDescInferFn(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const int32_t spatial_ndim = x_shape.NumAxes() - 2;
  std::string data_format = ctx->Attr<std::string>("data_format");
  std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
  const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
  const int32_t idx_offset = IdxOffset(data_format);
  const size_t c_dim = data_format == "channels_first" ? 1 : spatial_ndim + 1;

  CHECK_EQ_OR_RETURN(spatial_ndim, 2);  // only support 4-D tensor now.
  CHECK_EQ_OR_RETURN(padding.size(), spatial_ndim);
  for (int32_t pad : padding) { CHECK_GE_OR_RETURN(pad, 0); }
  CHECK_EQ_OR_RETURN(kernel_size.size(), spatial_ndim);
  for (int32_t kernel : kernel_size) { CHECK_GT_OR_RETURN(kernel, 0); }
  CHECK_EQ_OR_RETURN(strides.size(), spatial_ndim);
  for (int32_t stride : strides) { CHECK_GT_OR_RETURN(stride, 0); }
  CHECK_EQ_OR_RETURN(dilation_rate.size(), spatial_ndim);
  for (int32_t dilation : dilation_rate) { CHECK_GE_OR_RETURN(dilation, 1); }

  std::vector<int64_t> dhw_shape(spatial_ndim);
  for (int32_t i = 0; i < spatial_ndim; ++i) {
    dhw_shape[i] =
        (x_shape.At(idx_offset + i) + 2 * padding[i] - dilation_rate[i] * (kernel_size[i] - 1) - 1)
            / strides[i]
        + 1;
  }

  DimVector y_shape(3);
  y_shape.at(0) = x_shape.At(0);
  y_shape.at(1) =
      x_shape.At(c_dim)
      * std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<int>());
  y_shape.at(2) = std::accumulate(dhw_shape.begin(), dhw_shape.end(), 1, std::multiplies<int>());

  ctx->SetOutputShape("y", 0, Shape(y_shape));
  return Maybe<void>::Ok();
}

Maybe<void> SetUnfoldDTypeFn(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetUnfoldSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();

  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 1).Split(user_op::OpArg("y", 0), 1).Build();
  return Maybe<void>::Ok();
}

Maybe<void> FoldTensorDescInferFn(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const int32_t spatial_ndim = x_shape.NumAxes() - 1;  // (n, c*K*K, h*w)

  std::string data_format = ctx->Attr<std::string>("data_format");
  std::vector<int32_t> output_size = ctx->Attr<std::vector<int32_t>>("output_size");
  std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
  const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
  const size_t c_dim = data_format == "channels_first" ? 1 : spatial_ndim;
  const size_t length_dim = data_format == "channels_first" ? spatial_ndim : 1;

  const int32_t input_planes = x_shape.At(c_dim);
  const int32_t input_length = x_shape.At(length_dim);

  CHECK_EQ_OR_RETURN(spatial_ndim, 2);  // only support 4-D tensor now.
  CHECK_EQ_OR_RETURN(output_size.size(), spatial_ndim);
  CHECK_EQ_OR_RETURN(padding.size(), spatial_ndim);
  for (int32_t pad : padding) { CHECK_GE_OR_RETURN(pad, 0); }
  CHECK_EQ_OR_RETURN(kernel_size.size(), spatial_ndim);
  for (int32_t kernel : kernel_size) { CHECK_GT_OR_RETURN(kernel, 0); }
  CHECK_EQ_OR_RETURN(strides.size(), spatial_ndim);
  for (int32_t stride : strides) { CHECK_GT_OR_RETURN(stride, 0); }
  CHECK_EQ_OR_RETURN(dilation_rate.size(), spatial_ndim);
  for (int32_t dilation : dilation_rate) { CHECK_GE_OR_RETURN(dilation, 1); }

  CHECK_EQ_OR_RETURN(input_planes % (kernel_size[0] * kernel_size[1]),
                     0);  // C*K*K should be divided by K*K

  const int32_t output_height =
      (output_size[0] + 2 * padding[0] - dilation_rate[0] * (kernel_size[0] - 1) - 1) / strides[0]
      + 1;
  const int32_t output_width =
      (output_size[1] + 2 * padding[1] - dilation_rate[1] * (kernel_size[1] - 1) - 1) / strides[1]
      + 1;
  CHECK_EQ_OR_RETURN(output_height * output_width, input_length);  // input_length == OH*OW

  DimVector y_shape(4);
  y_shape.at(0) = x_shape.At(0);
  y_shape.at(1) = input_planes / (kernel_size[0] * kernel_size[1]);
  y_shape.at(2) = output_size[0];
  y_shape.at(3) = output_size[1];

  ctx->SetOutputShape("y", 0, Shape(y_shape));
  return Maybe<void>::Ok();
}

Maybe<void> FoldDTypeFn(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetFoldSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> UnfoldOp::GetSbp(user_op::SbpContext* ctx) { return GetUnfoldSbpFn(ctx); }
/*static*/ Maybe<void> UnfoldOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return UnfoldTensorDescInferFn(ctx);
}
/*static*/ Maybe<void> UnfoldOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UnfoldOp::InferDataType(user_op::InferContext* ctx) {
  return SetUnfoldDTypeFn(ctx);
}

/*static*/ Maybe<void> FoldOp::GetSbp(user_op::SbpContext* ctx) { return GetFoldSbpFn(ctx); }
/*static*/ Maybe<void> FoldOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return FoldTensorDescInferFn(ctx);
}
/*static*/ Maybe<void> FoldOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> FoldOp::InferDataType(user_op::InferContext* ctx) {
  return FoldDTypeFn(ctx);
}

}  // namespace oneflow
