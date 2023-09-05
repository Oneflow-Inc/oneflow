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

/* static */ Maybe<void> FusedActivationMinMaxObserverOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  CHECK_OR_RETURN(ctx->Attr<bool>("per_layer_quantization"))
      << "activation min_max_observer only support per-layer quantization";
  const Shape& weight_scale_shape = ctx->InputShape("weight_scale", 0);

  ctx->SetOutputShape("in_scale", 0, Shape({1}));
  ctx->SetOutputShape("in_zero_point", 0, Shape({1}));
  ctx->SetOutputShape("out_scale", 0, {weight_scale_shape.Count(0)});
  ctx->SetOutputShape("out_bias", 0, {weight_scale_shape.Count(0)});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedActivationMinMaxObserverOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedActivationMinMaxObserverOp::GetSbp(user_op::SbpContext* ctx) {
  // NOTE(Liang Depeng): input needs to be broadcast in order to accurately calculate the
  // global scale and zero_point
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedActivationMinMaxObserverOp::CheckAttr(
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

/* static */ Maybe<void> FusedActivationMinMaxObserverOp::InferDataType(
    user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("weight_scale", 0), DataType::kFloat)
      << "weight_scale dtype should be float";
  CHECK_EQ_OR_RETURN(ctx->InputDType("weight_acc", 0), DataType::kFloat)
      << "weight_acc dtype should be float";

  DataType data_type = ctx->InputDType("in", 0);
  if (ctx->has_input("bias", 0)) { CHECK_EQ_OR_RETURN(data_type, ctx->InputDType("bias", 0)); }

  int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
  const std::string& quantization_formula = ctx->Attr<std::string>("quantization_formula");
  if (quantization_formula == "oneflow") {
    if (quantization_bit == 8) {
      ctx->SetOutputDType("in_zero_point", 0, DataType::kInt8);
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    ctx->SetOutputDType("in_zero_point", 0, data_type);
  }
  ctx->SetOutputDType("in_scale", 0, data_type);
  ctx->SetOutputDType("out_scale", 0, data_type);
  ctx->SetOutputDType("out_bias", 0, data_type);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
