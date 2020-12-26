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

REGISTER_USER_OP("generate_quantize_scale_for_weight")
    .Input("weight")
    .Output("scale")
    .Output("zero_point")
    // NOTE(Liang Depeng): quantize from float32 to "quantize_to_bit" bit signed or unsigned integer
    .Attr<int32_t>("quantize_to_bit", 8)
    // NOTE(Liang Depeng): "symmetric" or "affine": quantize to signed or unsigned integer
    .Attr<std::string>("quantizer_type", "symmetric")
    // NOTE(Liang Depeng): "true" or "false": per-layer or per-channel quantization
    .Attr<bool>("per_layer_quantization", true)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* weight_shape = ctx->Shape4ArgNameAndIndex("weight", 0);
      // NOTE(Liang Depeng): only support weights for 2D convolution and fully-connected layers.
      //                     And assume weight shape is (Cout, Cin, K, K) or (Cout, Cin) for 2D
      //                     convolution or fully-connected.
      CHECK_OR_RETURN(weight_shape->NumAxes() == 4 || weight_shape->NumAxes() == 2);

      if (ctx->Attr<bool>("per_layer_quantization") == true) {
        *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({1});
        *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({1});
      } else {
        *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({weight_shape->At(0)});
        *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({weight_shape->At(0)});
      }

      *ctx->Dtype4ArgNameAndIndex("scale", 0) = *ctx->Dtype4ArgNameAndIndex("weight", 0);
      *ctx->Dtype4ArgNameAndIndex("zero_point", 0) = *ctx->Dtype4ArgNameAndIndex("weight", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* weight = GetInputArgModifierFn("weight", 0);
      weight->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto ClearBatchAxis = [ctx](const std::string& name) {
        if (ctx->user_op_conf().has_output(name, 0)) {
          ctx->BatchAxis4ArgNameAndIndex(name, 0)->clear_value();
        }
      };
      ClearBatchAxis("scale");
      ClearBatchAxis("zero_point");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO(Liang Depeng): refer to reduce_max op
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      int32_t quantize_to_bit = op_conf.attr<int32_t>("quantize_to_bit");
      CHECK_GT_OR_RETURN(quantize_to_bit, 0);
      CHECK_LE_OR_RETURN(quantize_to_bit, 8);

      std::string quantizer_type = op_conf.attr<std::string>("quantizer_type");
      CHECK_OR_RETURN(quantizer_type == "symmetric" || quantizer_type == "affine");
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
