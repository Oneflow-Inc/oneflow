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

/* static */ Maybe<void> DynamicQuantizationOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(ctx->Attr<bool>("per_layer_quantization"))
      << "dynamic quantization only supports per-layer quantization";
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputShape("scale", 0, Shape({1}));
  ctx->SetOutputShape("zero_point", 0, Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DynamicQuantizationOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DynamicQuantizationOp::GetSbp(user_op::SbpContext* ctx) {
  // NOTE(Liang Depeng): input needs to be broadcast in order to accurately calculate the
  // global scale and zero_point
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DynamicQuantizationOp::CheckAttr(
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

/* static */ Maybe<void> DynamicQuantizationOp::InferDataType(user_op::InferContext* ctx) {
  int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
  const std::string& quantization_formula = ctx->Attr<std::string>("quantization_formula");
  if (quantization_formula == "oneflow") {
    if (quantization_bit == 8) {
      ctx->SetOutputDType("out", 0, DataType::kInt8);
      ctx->SetOutputDType("zero_point", 0, DataType::kInt8);
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    OF_UNIMPLEMENTED();
  }
  ctx->SetOutputDType("scale", 0, DataType::kFloat);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
