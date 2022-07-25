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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> SamePaddingOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("x_like", 0).shape().NumAxes();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
  const int32_t channel_idx = ChannelIdx(data_format, num_axes);
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), channel_idx)
      .Split(user_op::OpArg("y", 0), channel_idx)
      .Build();
  ctx->NewBuilder().PartialSum(user_op::OpArg("x", 0)).PartialSum(user_op::OpArg("y", 0)).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SamePaddingOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
  *y_desc->mut_shape() = x_desc.shape();
  *y_desc->mut_is_dynamic() = x_desc.is_dynamic();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
  const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
  const size_t idx_offset = IdxOffset(data_format);
  const int32_t num_spatial_dims = x_desc.shape().NumAxes() - 2;
  CHECK_EQ_OR_RETURN(num_spatial_dims, kernel_size.size())
      << Error::RuntimeError()
      << "The dimension of x tensor must be equal to the size of kernel_size array plus 2, "
      << "but got " << num_spatial_dims << " and " << kernel_size.size();
  CHECK_EQ_OR_RETURN(num_spatial_dims, strides.size())
      << Error::RuntimeError()
      << "The dimension of x tensor must be equal to the size of strides array plus 2, "
      << "but got " << num_spatial_dims << " and " << strides.size();
  CHECK_EQ_OR_RETURN(num_spatial_dims, dilation_rate.size())
      << Error::RuntimeError()
      << "The dimension of x tensor must be equal to the size of dilation_rate array plus 2, "
      << "but got " << num_spatial_dims << " and " << dilation_rate.size();
  DimVector y_dim_vec(x_desc.shape().dim_vec());
  for (int32_t i = 0; i < num_spatial_dims; ++i) {
    int32_t padding_small = 0;
    int32_t padding_large = 0;
    JUST(CalcSamePadding(x_desc.shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                         strides.at(i), &padding_small, &padding_large));
    y_dim_vec[idx_offset + i] = x_desc.shape().At(idx_offset + i) + padding_small + padding_large;
  }
  *y_desc->mut_shape() = Shape(y_dim_vec);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SamePaddingOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SamePaddingOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SamePaddingGradOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("x_like", 0).shape().NumAxes();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  ctx->NewBuilder()
      .Split(user_op::OpArg("x_like", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  const int32_t channel_idx = ChannelIdx(data_format, num_axes);
  ctx->NewBuilder()
      .Split(user_op::OpArg("x_like", 0), channel_idx)
      .Split(user_op::OpArg("dy", 0), channel_idx)
      .Split(user_op::OpArg("dx", 0), channel_idx)
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x_like", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x_like", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x_like", 0))
      .Broadcast(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SamePaddingGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("dx", 0) = ctx->InputShape("x_like", 0);
  *ctx->MutOutputIsDynamic("dx", 0) = ctx->InputIsDynamic("x_like", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SamePaddingGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SamePaddingGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("x_like", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("same_padding")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const std::string& padding = op.attr<std::string>("padding");
        const std::string& data_format = op.attr<std::string>("data_format");
        const auto& kernel_size = op.attr<std::vector<int32_t>>("kernel_size");
        const auto& strides = op.attr<std::vector<int32_t>>("strides");
        const auto& dilation_rate = op.attr<std::vector<int32_t>>("dilation_rate");
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("same_padding_grad")
                .Input("x_like", op.input("x", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr<std::string>("padding", padding)
                .Attr<std::string>("data_format", data_format)
                .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
                .Attr<std::vector<int32_t>>("strides", strides)
                .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
