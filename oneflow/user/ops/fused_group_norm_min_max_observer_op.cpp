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
#include <cstdint>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

}  // namespace

/* static */ Maybe<void> FusedGroupNormMinMaxObserverOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const bool affine = ctx->Attr<bool>("affine");

  const std::string& data_format = ctx->Attr<std::string>("data_format");
  int64_t channel_size = 0;
  if (data_format == "channels_first") {
    channel_size = x.shape().At(1);
  } else if (data_format == "channels_last") {
    channel_size = x.shape().back();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }

  // check input shape
  CHECK_GT_OR_RETURN(x.shape().NumAxes(), 2)
      << "fused_group_norm_min_max_observer: input should have at least 2 dims, got "
      << x.shape().NumAxes() << " dims";

  // check num_group
  int32_t num_groups = ctx->Attr<int32_t>("num_groups");
  CHECK_EQ_OR_RETURN(channel_size % num_groups, 0) << "Channels should be divisble by num_groups. ";

  // check gamma and beta size
  if (affine) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.shape().elem_cnt(), channel_size)
        << "The size of `gamma` must equal to channel_size, expected " << channel_size
        << " but got " << gamma.shape().elem_cnt();
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.shape().elem_cnt(), channel_size)
        << "The size of `beta` must equal to channel_size, expected " << channel_size << " but got "
        << beta.shape().elem_cnt();
  }

  CHECK_OR_RETURN(ctx->Attr<bool>("per_layer_quantization"))
      << "dynamic quantization only supports per-layer quantization";
  ctx->SetOutputShape("y", 0, x.shape());
  ctx->SetOutputShape("y_scale", 0, Shape({1}));
  ctx->SetOutputShape("y_zero_point", 0, Shape({1}));

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedGroupNormMinMaxObserverOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedGroupNormMinMaxObserverOp::GetSbp(user_op::SbpContext* ctx) {
  // dynamic quantization only supports broadcast
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedGroupNormMinMaxObserverOp::CheckAttr(
    const user_op::UserOpDefWrapper& def, const user_op::UserOpConfWrapper& op_conf) {
  int32_t quantization_bit = op_conf.attr<int32_t>("quantization_bit");
  CHECK_GT_OR_RETURN(quantization_bit, 1);
  CHECK_LE_OR_RETURN(quantization_bit, 8);

  std::string quantization_scheme = op_conf.attr<std::string>("quantization_scheme");
  CHECK_OR_RETURN(quantization_scheme == "symmetric" || quantization_scheme == "affine");

  std::string quantization_formula = op_conf.attr<std::string>("quantization_formula");
  CHECK_OR_RETURN(quantization_formula == "google" || quantization_formula == "cambricon"
                  || quantization_formula == "oneflow");
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedGroupNormMinMaxObserverOp::InferDataType(user_op::InferContext* ctx) {
  const bool affine = ctx->Attr<bool>("affine");
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  if (affine) {
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(beta.data_type());
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(gamma.data_type());
  }
  ctx->SetOutputDType("y", 0, x.data_type());
  ctx->SetOutputDType("y_scale", 0, DataType::kFloat);

  int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
  const std::string& quantization_formula = ctx->Attr<std::string>("quantization_formula");
  if (quantization_formula == "oneflow") {
    if (quantization_bit == 8) {
      ctx->SetOutputDType("y_zero_point", 0, DataType::kInt8);
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
