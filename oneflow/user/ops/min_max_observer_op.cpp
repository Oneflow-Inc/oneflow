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

/* static */ Maybe<void> MinMaxObserverOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);

  if (ctx->Attr<std::string>("quantization_formula") == "google") {
    if (ctx->Attr<bool>("per_layer_quantization") == true) {
      ctx->SetOutputShape("scale", 0, Shape({1}));
      ctx->SetOutputShape("zero_point", 0, Shape({1}));
    } else {
      // NOTE(Liang Depeng): For now per-channel quantization only support axis 0
      ctx->SetOutputShape("scale", 0, Shape({in_shape.At(0)}));
      ctx->SetOutputShape("zero_point", 0, Shape({in_shape.At(0)}));
    }
  } else {  // quantization_formula == "cambricon"
    ctx->SetOutputShape("scale", 0, Shape({1}));
    ctx->SetOutputShape("zero_point", 0, Shape({1}));
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MinMaxObserverOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MinMaxObserverOp::GetSbp(user_op::SbpContext* ctx) {
  // NOTE(Liang Depeng): input needs to be broadcast in order to accurately calculate the
  // global scale and zero_point
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MinMaxObserverOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
  CHECK_OR_RETURN(in != nullptr);
  in->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MinMaxObserverOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                     const user_op::UserOpConfWrapper& op_conf) {
  int32_t quantization_bit = op_conf.attr<int32_t>("quantization_bit");
  CHECK_GT_OR_RETURN(quantization_bit, 1);
  CHECK_LE_OR_RETURN(quantization_bit, 8);

  std::string quantization_scheme = op_conf.attr<std::string>("quantization_scheme");
  CHECK_OR_RETURN(quantization_scheme == "symmetric" || quantization_scheme == "affine");

  std::string quantization_formula = op_conf.attr<std::string>("quantization_formula");
  CHECK_OR_RETURN(quantization_formula == "google" || quantization_formula == "cambricon");
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MinMaxObserverOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("scale", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("zero_point", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
