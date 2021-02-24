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

namespace oneflow {

namespace {

REGISTER_USER_OP("min_max_observer")
    .Input("in")
    .Output("scale")
    .Output("zero_point")
    // NOTE(Liang Depeng): "google" or "cambricon"
    .Attr<std::string>("quantization_formula", "google")
    // NOTE(Liang Depeng): quantize from float32 to "quantization_bit" bit signed or unsigned
    // integer
    .Attr<int32_t>("quantization_bit", 8)
    // NOTE(Liang Depeng): "symmetric" or "affine": quantize to signed or unsigned integer
    .Attr<std::string>("quantization_scheme", "symmetric")
    // NOTE(Liang Depeng): "true" or "false": per-layer or per-channel quantization.
    .Attr<bool>("per_layer_quantization", true)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);

      if (ctx->Attr<std::string>("quantization_formula") == "google") {
        if (ctx->Attr<bool>("per_layer_quantization") == true) {
          *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({1});
          *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({1});
        } else {
          // NOTE(Liang Depeng): For now per-channel quantization only support axis 0
          *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({in_shape->At(0)});
          *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({in_shape->At(0)});
        }
      } else {  // quantization_formula == "cambricon"
        *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({1});
        *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({1});
      }

      *ctx->Dtype4ArgNameAndIndex("scale", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("zero_point", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
      CHECK(in != nullptr);
      in->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // NOTE(Liang Depeng): input needs to be broadcast in order to accurately calculate the
      // global scale and zero_point
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      int32_t quantization_bit = op_conf.attr<int32_t>("quantization_bit");
      CHECK_GT_OR_RETURN(quantization_bit, 1);
      CHECK_LE_OR_RETURN(quantization_bit, 8);

      std::string quantization_scheme = op_conf.attr<std::string>("quantization_scheme");
      CHECK_OR_RETURN(quantization_scheme == "symmetric" || quantization_scheme == "affine");

      std::string quantization_formula = op_conf.attr<std::string>("quantization_formula");
      CHECK_OR_RETURN(quantization_formula == "google" || quantization_formula == "cambricon");
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
