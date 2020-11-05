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

REGISTER_USER_OP("generate_quantize_scale_for_activation")
    .Input("activation")
    .Input("moving_max")
    .Input("moving_min")
    .Output("scale")
    .Output("zero_point")
    // NOTE(Liang Depeng): quantize from float32 to "quantize_to_bit" bit signed or unsigned integer
    .Attr<int32_t>("quantize_to_bit", 8)
    // NOTE(Liang Depeng): "symmetric" or "affine": quantize to signed or unsigned integer
    .Attr<std::string>("quantizer_type", "symmetric")
    // NOTE(Liang Depeng): smoothing parameter for exponential moving averages operation
    .Attr<float>("momentum", 0.95)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* moving_max_shape = ctx->Shape4ArgNameAndIndex("moving_max", 0);
      Shape* moving_min_shape = ctx->Shape4ArgNameAndIndex("moving_min", 0);
      // NOTE(Liang Depeng): activation only support per-layer quantization
      CHECK_OR_RETURN(moving_max_shape->NumAxes() == 1 && moving_max_shape->At(0) == 1);
      CHECK_OR_RETURN(moving_min_shape->NumAxes() == 1 && moving_min_shape->At(0) == 1);

      *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({1});
      *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({1});
      *ctx->Dtype4ArgNameAndIndex("scale", 0) = *ctx->Dtype4ArgNameAndIndex("activation", 0);
      *ctx->Dtype4ArgNameAndIndex("zero_point", 0) = *ctx->Dtype4ArgNameAndIndex("activation", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* activation = GetInputArgModifierFn("activation", 0);
      CHECK(activation != nullptr);
      activation->set_requires_grad(false);

      user_op::InputArgModifier* moving_max = GetInputArgModifierFn("moving_max", 0);
      CHECK(moving_max != nullptr);
      moving_max->set_requires_grad(false);
      moving_max->set_is_mutable(true);

      user_op::InputArgModifier* moving_min = GetInputArgModifierFn("moving_min", 0);
      CHECK(moving_min != nullptr);
      moving_min->set_requires_grad(false);
      moving_min->set_is_mutable(true);
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
      // TODO(Liang Depeng): refer to reduce_max and normalization op
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      int32_t quantize_to_bit = op_conf.attr<int32_t>("quantize_to_bit");
      CHECK_GT_OR_RETURN(quantize_to_bit, 1);
      CHECK_LE_OR_RETURN(quantize_to_bit, 8);

      std::string quantizer_type = op_conf.attr<std::string>("quantizer_type");
      CHECK_OR_RETURN(quantizer_type == "symmetric" || quantizer_type == "affine");
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
