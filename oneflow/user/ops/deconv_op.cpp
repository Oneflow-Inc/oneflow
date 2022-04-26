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
Maybe<void> InferTensorDesc4DeConv(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);

  CHECK_EQ_OR_RETURN(NDims + 2, in.shape().NumAxes());

  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  CHECK_EQ_OR_RETURN(NDims, kernel_size.size());
  const int32_t filters = ctx->Attr<int32_t>("filters");
  size_t idx_offset = IdxOffset(data_format);
  int32_t groups = ctx->Attr<int32_t>("groups");
  {
    const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const auto& output_padding = ctx->Attr<std::vector<int32_t>>("output_padding");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    CHECK_EQ_OR_RETURN(NDims, dilation_rate.size());
    CHECK_EQ_OR_RETURN(NDims, strides.size());
    CHECK_EQ_OR_RETURN(NDims, output_padding.size());

    user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
    DimVector out_shape(NDims + 2);
    out_shape.at(0) = in.shape().At(0);
    const size_t c_dim = data_format == "channels_first" ? 1 : NDims + 1;
    out_shape.at(c_dim) = filters;
    for (int32_t i = 0; i < NDims; ++i) {
      int32_t effective_filter_size = (kernel_size.at(i) - 1) * dilation_rate.at(i) + 1;
      out_shape.at(idx_offset + i) = (in.shape().At(idx_offset + i) - 1) * strides.at(i)
                                     - 2 * padding_before.at(i) + output_padding.at(i)
                                     + effective_filter_size;
    }
    if (in.shape().At(0) != 0) {
      for (int i = 0; i < out_shape.size(); i++) {
        CHECK_GT_OR_RETURN(out_shape[i], 0)
            << "RuntimeError: Given input size per channel: (" << Shape(in.shape())
            << "). Calculated output size per channel: (" << Shape(out_shape)
            << "). Output size is too small";
      }
    }
    *out->mut_is_dynamic() = in.is_dynamic();
    *out->mut_shape() = Shape(out_shape);
  }

  {
    DimVector weight_shape(in.shape().dim_vec());
    if (data_format == "channels_first") {
      weight_shape.at(0) = in.shape().At(1);
      weight_shape.at(1) = filters / groups;
    } else if (data_format == "channels_last") {
      weight_shape.at(0) = in.shape().At(NDims + 1);
      weight_shape.at(NDims + 1) = filters / groups;
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    for (size_t i = 0; i < NDims; ++i) { weight_shape.at(idx_offset + i) = kernel_size.at(i); }
    const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight.shape(), Shape(weight_shape));
  }

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType_(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures4DeConv(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("in", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();

  return Maybe<void>::Ok();
}

template<size_t NDims>
Maybe<void> CheckAttr_(const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) {
  bool is_checked = true;
  std::stringstream err;
  err << "Illegal value for " << conf.op_type_name() << " op " << conf.op_name() << ": ";

  const std::string& data_format = conf.attr<std::string>("data_format");
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

Maybe<void> GenerateBackwardOpConf4DeConv(const user_op::UserOpWrapper& op,
                                          user_op::AddOpFn AddOp) {
  const std::string& data_format = op.attr<std::string>("data_format");
  const auto& padding_before = op.attr<std::vector<int32_t>>("padding_before");
  const auto& kernel_size = op.attr<std::vector<int32_t>>("kernel_size");
  const auto& strides = op.attr<std::vector<int32_t>>("strides");
  const auto& dilation_rate = op.attr<std::vector<int32_t>>("dilation_rate");
  const Shape& weight_shape = op.TensorDesc4ArgNameAndIndex("weight", 0).shape();
  int32_t groups = op.attr<int32_t>("groups");

  const int32_t ndims = kernel_size.size();
  CHECK_EQ_OR_RETURN(ndims, strides.size());
  CHECK_EQ_OR_RETURN(ndims, dilation_rate.size());

  if (op.NeedGenGradTensor4OpInput("weight", 0)) {
    auto filter_grad_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-" + op.op_name() + "-FilterGrad")
            .Op("conv_filter_grad")
            .Input("dy", op.input("in", 0))
            .Input("x", op.GetGradTensorWithOpOutput("out", 0))
            .Output("filter_diff")
            .Attr<int32_t>("num_spatial_dims", ndims)
            .Attr<std::vector<int32_t>>("padding_before", padding_before)
            .Attr<std::string>("data_format", data_format)
            .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
            .Attr<std::vector<int32_t>>("strides", strides)
            .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
            .Attr<int32_t>("groups", groups)
            .Build();
    op.BindGradTensorWithOpInput(filter_grad_op.output("filter_diff", 0), "weight", 0);
    AddOp(filter_grad_op);
  }

  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    std::string ndims_str = std::to_string(ndims);
    auto data_grad_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-" + op.op_name() + "-DataGrad")
            .Op("conv" + ndims_str + "d")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Input("weight", op.input("weight", 0))
            .Output("out")
            .Attr<int32_t>("filters", weight_shape.At(0))
            .Attr<std::string>("data_format", data_format)
            .Attr<std::vector<int32_t>>("padding_before", padding_before)
            .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
            .Attr<std::vector<int32_t>>("strides", strides)
            .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
            .Attr<int32_t>("groups", groups)
            .Build();
    op.BindGradTensorWithOpInput(data_grad_op.output("out", 0), "in", 0);
    AddOp(data_grad_op);
  }
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> Deconv1DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4DeConv<1>(ctx);
}

/*static*/ Maybe<void> Deconv1DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Deconv1DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4DeConv(ctx);
}

/* static */ Maybe<void> Deconv1DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                               const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<1>(def, conf);
}

/* static */ Maybe<void> Deconv1DOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType_(ctx);
}

/* static */ Maybe<void> Deconv2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4DeConv<2>(ctx);
}

/*static*/ Maybe<void> Deconv2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Deconv2DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4DeConv(ctx);
}

/* static */ Maybe<void> Deconv2DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                               const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<2>(def, conf);
}

/* static */ Maybe<void> Deconv2DOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType_(ctx);
}

/* static */ Maybe<void> Deconv3DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4DeConv<3>(ctx);
}

/*static*/ Maybe<void> Deconv3DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Deconv3DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures4DeConv(ctx);
}

/* static */ Maybe<void> Deconv3DOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                               const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_<3>(def, conf);
}

/* static */ Maybe<void> Deconv3DOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType_(ctx);
}

REGISTER_USER_OP_GRAD("deconv1d").SetGenBackwardOpConfFn(GenerateBackwardOpConf4DeConv);
REGISTER_USER_OP_GRAD("deconv2d").SetGenBackwardOpConfFn(GenerateBackwardOpConf4DeConv);
REGISTER_USER_OP_GRAD("deconv3d").SetGenBackwardOpConfFn(GenerateBackwardOpConf4DeConv);

}  // namespace oneflow
