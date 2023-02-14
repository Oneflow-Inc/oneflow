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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

template<size_t NDims>
Maybe<void> InferTensorDesc4Conv(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(NDims + 2, in.shape().NumAxes())
      << "Conv" << NDims << "D op's input shape ndim should equal to " << NDims + 2
      << " ,but got: " << in.shape().NumAxes();

  auto data_format = ctx->Attr<std::string>("data_format");
  auto kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  CHECK_EQ_OR_RETURN(NDims, kernel_size.size());
  int32_t filters = ctx->Attr<int32_t>("filters");
  size_t idx_offset = IdxOffset(data_format);
  {
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    auto dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    auto strides = ctx->Attr<std::vector<int32_t>>("strides");
    CHECK_EQ_OR_RETURN(NDims, dilation_rate.size());
    CHECK_EQ_OR_RETURN(NDims, strides.size());
    CHECK_EQ_OR_RETURN(NDims, padding_before.size());

    user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
    DimVector out_shape(NDims + 2);
    out_shape.at(0) = in.shape().At(0);
    const size_t c_dim = data_format == "channels_first" ? 1 : NDims + 1;
    out_shape.at(c_dim) = filters;
    for (int32_t i = 0; i < NDims; ++i) {
      JUST(CalcConvOut(in.shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                       strides.at(i), padding_before.at(i), &out_shape.at(idx_offset + i)));
    }
    out->set_is_dynamic(in.is_dynamic());
    out->set_shape(Shape(out_shape));
  }

  {
    int32_t groups = ctx->Attr<int32_t>("groups");
    CHECK_GT_OR_RETURN(groups, 0);
    CHECK_LE_OR_RETURN(groups, filters);
    CHECK_EQ_OR_RETURN(filters % groups, 0);

    DimVector weight_shape(in.shape().dim_vec());
    weight_shape.at(0) = filters;
    if (data_format == "channels_first") {
      CHECK_LE_OR_RETURN(groups, weight_shape.at(1));
      CHECK_EQ_OR_RETURN(weight_shape.at(1) % groups, 0);
      weight_shape.at(1) = weight_shape.at(1) / groups;
    } else if (data_format == "channels_last") {
      CHECK_LE_OR_RETURN(groups, weight_shape.at(NDims + 1));
      CHECK_EQ_OR_RETURN(weight_shape.at(NDims + 1) % groups, 0);
      weight_shape.at(NDims + 1) = weight_shape.at(NDims + 1) / groups;
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    for (size_t i = 0; i < NDims; ++i) { weight_shape.at(idx_offset + i) = kernel_size.at(i); }

    const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight.shape(), Shape(weight_shape));
  }

  bool has_bias = ctx->has_input("bias", 0);
  if (has_bias) {
    const user_op::TensorDesc& bias = ctx->InputTensorDesc("bias", 0);
    CHECK_EQ_OR_RETURN(bias.shape(), Shape({filters}));
  }

  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures4Conv(user_op::SbpContext* ctx) {
  bool has_bias = false;
  for (const auto& pair : ctx->inputs()) {
    if (pair.first == "bias") {
      CHECK_EQ_OR_RETURN(0, pair.second);
      has_bias = true;
      break;
    }
  }

  if (has_bias) {
    ctx->NewBuilder()
        .Split(ctx->inputs(), 0)
        .Split(user_op::OpArg("in", 0), 0)
        .Broadcast(user_op::OpArg("weight", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  } else {
    ctx->NewBuilder()
        .Split(ctx->inputs(), 0)
        .Split(user_op::OpArg("in", 0), 0)
        .Broadcast(user_op::OpArg("weight", 0))
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*
Example for conv2d:

ComputationCost
= ((k*k + k*k-1)*c + c-1 + bias?1:0) * out_channel * out_width * out_height * batch_size
= (2*k*k*c - 1 + bias?1:0) * out_channel * out_width * out_height * batch_size
≈ 2*k*k*c * out_channel * out_width * out_height * batch_size
*/
Maybe<double> ConvComputationCost(user_op::ComputeComplexityFnContext* ctx) {
  const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::string data_format = ctx->Attr<std::string>("data_format");
  const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const size_t c_dim = data_format == "channels_first" ? 1 : in->shape().NumAxes() - 1;
  const int32_t c = in->shape().At(c_dim);
  const user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  double cost =
      std::accumulate(kernel_size.begin(), kernel_size.end(), 1.0, std::multiplies<double>());
  cost = cost * 2 * c;
  cost *= std::accumulate(out->shape().dim_vec().begin(), out->shape().dim_vec().end(), 1.0,
                          std::multiplies<double>());

  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  const auto& nd_sbp_out = ctx->NdSbp4ArgNameAndIndex("out", 0);
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp_out.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp_out.sbp_parallel(dim_sbp).has_split_parallel()) {
      cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return cost;
}

template<size_t NDims>
Maybe<void> CheckAttr_(const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) {
  bool is_checked = true;
  std::stringstream err;
  err << "Illegal value for " << conf.op_type_name() << " op " << conf.op_name() << ": ";

  const auto& data_format = conf.attr<std::string>("data_format");
  if (!(data_format == "channels_first" || data_format == "channels_last")) {
    err << " data_format:" << data_format;
    is_checked = false;
  }

  if (NDims != 0) {
    const auto& padding_before = conf.attr<std::vector<int32_t>>("padding_before");
    if (padding_before.size() != NDims) {
      err << " padding_before: number of element is " << padding_before.size();
      is_checked = false;
    }

    const auto& kernel_size = conf.attr<std::vector<int32_t>>("kernel_size");
    if (kernel_size.size() != NDims) {
      err << " kernel_size: number of element is " << kernel_size.size();
      is_checked = false;
    }

    const auto& strides = conf.attr<std::vector<int32_t>>("strides");
    if (strides.size() != NDims) {
      err << " strides: number of element is " << strides.size();
      is_checked = false;
    }

    const auto& dilation_rate = conf.attr<std::vector<int32_t>>("dilation_rate");
    if (dilation_rate.size() != NDims) {
      err << " dilation_rate: number of element is " << dilation_rate.size();
      is_checked = false;
    }
  }

  if (is_checked) {
    return Maybe<void>::Ok();
  } else {
    return oneflow::Error::CheckFailedError() << err.str();
  }
}

}  // namespace

/* static */ Maybe<void> Conv1DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Conv<1>(ctx);
}

/*static*/ Maybe<void> Conv1DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Conv1DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4Conv(ctx);
}

/* static */ Maybe<double> Conv1DOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ConvComputationCost(ctx);
}

/* static */ Maybe<void> Conv1DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                             const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<1>(def, conf);
}

/* static */ Maybe<void> Conv1DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Conv2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Conv<2>(ctx);
}

/*static*/ Maybe<void> Conv2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Conv2DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4Conv(ctx);
}

/* static */ Maybe<double> Conv2DOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ConvComputationCost(ctx);
}

/* static */ Maybe<void> Conv2DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                             const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<2>(def, conf);
}

/* static */ Maybe<void> Conv2DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Conv3DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Conv<3>(ctx);
}

/*static*/ Maybe<void> Conv3DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Conv3DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4Conv(ctx);
}

/* static */ Maybe<double> Conv3DOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ConvComputationCost(ctx);
}

/* static */ Maybe<void> Conv3DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                             const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<3>(def, conf);
}

/* static */ Maybe<void> Conv3DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ConvDataGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x_like = ctx->InputTensorDesc("x_like", 0);
  const int32_t num_spatial_dims = ctx->Attr<int32_t>("num_spatial_dims");
  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy.shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x_like.shape().NumAxes(), num_spatial_dims + 2);
  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), x_like.shape());
  }
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x_like", 0));
  ctx->SetOutputIsDynamic("dx", 0, ctx->InputIsDynamic("x_like", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConvDataGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ConvDataGradOp::GetSbp(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("dy", 0);
  split_args.emplace_back("x_like", 0);
  split_args.emplace_back("dx", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    split_args.emplace_back("_add_to_output", 0);
  }
  ctx->NewBuilder().Split(split_args, 0).Broadcast(user_op::OpArg("filter", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ConvDataGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                   const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<0>(def, conf);
}

/* static */ Maybe<void> ConvDataGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x_like = ctx->InputTensorDesc("x_like", 0);
  CHECK_EQ_OR_RETURN(x_like.data_type(), dy.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(dy.data_type()) << ", but got "
      << DataType_Name(x_like.data_type());
  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.data_type(), x_like.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(add_to_output.data_type())
        << ", but got " << DataType_Name(x_like.data_type());
  }
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x_like", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<double> ConvDataGradOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const user_op::TensorDesc* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
  const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const size_t c_dim =
      ctx->Attr<std::string>("data_format") == "channels_first" ? 1 : dy->shape().NumAxes() - 1;

  double cost =
      std::accumulate(kernel_size.begin(), kernel_size.end(), 1.0, std::multiplies<double>())
      * std::accumulate(dx->shape().dim_vec().begin(), dx->shape().dim_vec().end(), 1.0,
                        std::multiplies<double>())
      * 2.0 * dy->shape().At(c_dim);

  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("dx", 0);
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp.sbp_parallel(dim_sbp).has_split_parallel()) {
      cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return cost;
}

/* static */ Maybe<void> ConvFilterGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);

  const int32_t num_spatial_dims = ctx->Attr<int32_t>("num_spatial_dims");
  const int32_t groups = ctx->Attr<int32_t>("groups");
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");

  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy.shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x.shape().NumAxes(), num_spatial_dims + 2);
  CHECK_GT_OR_RETURN(groups, 0);

  DimVector filter_diff_dim_vec;
  if (data_format == "channels_first") {
    CHECK_LE_OR_RETURN(groups, x.shape().At(1));
    CHECK_LE_OR_RETURN(groups, dy.shape().At(1));
    CHECK_EQ_OR_RETURN(x.shape().At(1) % groups, 0);
    CHECK_EQ_OR_RETURN(dy.shape().At(1) % groups, 0);
    filter_diff_dim_vec.emplace_back(dy.shape().At(1));
    filter_diff_dim_vec.emplace_back(x.shape().At(1) / groups);
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), kernel_size.cbegin(), kernel_size.cend());
  } else {
    CHECK_EQ_OR_RETURN("channels_last", data_format);
    CHECK_EQ_OR_RETURN(groups, 1);
    filter_diff_dim_vec.emplace_back(dy.shape().dim_vec().back());
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), kernel_size.cbegin(), kernel_size.cend());
    filter_diff_dim_vec.emplace_back(x.shape().dim_vec().back() / groups);
  }

  user_op::TensorDesc* filter_diff = ctx->MutOutputTensorDesc("filter_diff", 0);
  filter_diff->set_shape(Shape(filter_diff_dim_vec));
  filter_diff->set_is_dynamic(false);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConvFilterGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ConvFilterGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .PartialSum(user_op::OpArg("filter_diff", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ConvFilterGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                     const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<0>(def, conf);
}

/* static */ Maybe<void> ConvFilterGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  CHECK_EQ_OR_RETURN(x.data_type(), dy.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(dy.data_type()) << ", but got "
      << DataType_Name(x.data_type());
  user_op::TensorDesc* filter_diff = ctx->MutOutputTensorDesc("filter_diff", 0);
  filter_diff->set_data_type(x.data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<double> ConvFilterGradOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const size_t c_dim =
      ctx->Attr<std::string>("data_format") == "channels_first" ? 1 : x->shape().NumAxes() - 1;

  double cost =
      std::accumulate(kernel_size.begin(), kernel_size.end(), 1.0, std::multiplies<double>())
      * std::accumulate(dy->shape().dim_vec().begin(), dy->shape().dim_vec().end(), 1.0,
                        std::multiplies<double>())
      * 2.0 * x->shape().At(c_dim);

  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("dy", 0);
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp.sbp_parallel(dim_sbp).has_split_parallel()) {
      cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return cost;
}

/* static */ Maybe<void> ConvBiasGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* bias_diff = ctx->MutOutputTensorDesc("bias_diff", 0);

  int32_t num_spatial_dims = ctx->Attr<int32_t>("num_spatial_dims");
  std::string data_format = ctx->Attr<std::string>("data_format");

  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy.shape().NumAxes(), num_spatial_dims + 2);
  if (data_format == "channels_first") {
    bias_diff->set_shape(Shape({dy.shape().At(1)}));
  } else if (data_format == "channels_last") {
    bias_diff->set_shape(Shape({dy.shape().At(dy.shape().NumAxes() - 1)}));
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConvBiasGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ConvBiasGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .PartialSum(user_op::OpArg("bias_diff", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ConvBiasGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                   const user_op::UserOpConfWrapper& conf) {
  std::string data_format = conf.attr<std::string>("data_format");
  if (data_format == "channels_first" || data_format == "channels_last") {
    return Maybe<void>::Ok();
  }
  return oneflow::Error::CheckFailedError() << "Illegal value for " << conf.op_type_name() << " op "
                                            << conf.op_name() << ": data_format:" << data_format;
}

/* static */ Maybe<void> ConvBiasGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* bias_diff = ctx->MutOutputTensorDesc("bias_diff", 0);
  bias_diff->set_data_type(dy.data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<double> ConvBiasGradOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const std::string data_format = ctx->Attr<std::string>("data_format");
  double cost = std::accumulate(dy->shape().dim_vec().begin(), dy->shape().dim_vec().end(), 1.0,
                                std::multiplies<double>());
  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("dy", 0);
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp.sbp_parallel(dim_sbp).has_split_parallel()) {
      cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return cost;
}

}  // namespace oneflow
