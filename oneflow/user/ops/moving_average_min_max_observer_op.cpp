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

REGISTER_USER_OP("moving_average_min_max_observer")
    .Input("in")
    .Input("current_train_step")
    .Input("moving_max")  // NOTE(Liang Depeng): needs to be initialized as 0
    .Input("moving_min")  // NOTE(Liang Depeng): needs to be initialized as 0
    .Output("scale")
    .Output("zero_point")
    .Attr<bool>("training")
    // NOTE(Liang Depeng): "google" or "cambricon"
    .Attr<std::string>("quantization_formula", "google")
    .Attr<int64_t>("stop_update_after_iters")
    // NOTE(Liang Depeng): quantize from float32 to "quantization_bit" bit signed or unsigned
    // integer
    .Attr<int32_t>("quantization_bit", 8)
    // NOTE(Liang Depeng): "symmetric" or "affine": quantize to signed or unsigned integer
    .Attr<std::string>("quantization_scheme", "symmetric")
    // NOTE(Liang Depeng): smoothing parameter for exponential moving average operation
    .Attr<float>("momentum", 0.95)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* moving_max_shape = ctx->Shape4ArgNameAndIndex("moving_max", 0);
      Shape* moving_min_shape = ctx->Shape4ArgNameAndIndex("moving_min", 0);
      Shape* current_train_step = ctx->Shape4ArgNameAndIndex("current_train_step", 0);

      // NOTE(Liang Depeng): for now only support per-layer quantization
      // TODO(Liang Depeng): depthwise convolution support per-channel quantization
      CHECK_OR_RETURN(moving_max_shape->NumAxes() == 1 && moving_max_shape->At(0) == 1);
      CHECK_OR_RETURN(moving_min_shape->NumAxes() == 1 && moving_min_shape->At(0) == 1);

      CHECK_OR_RETURN(current_train_step->NumAxes() == 1 && current_train_step->At(0) == 1);

      *ctx->Shape4ArgNameAndIndex("scale", 0) = Shape({1});
      *ctx->Shape4ArgNameAndIndex("zero_point", 0) = Shape({1});
      *ctx->Dtype4ArgNameAndIndex("scale", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("zero_point", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
      CHECK(in != nullptr);
      in->set_requires_grad(false);

      user_op::InputArgModifier* current_train_step =
          GetInputArgModifierFn("current_train_step", 0);
      CHECK(current_train_step != nullptr);
      current_train_step->set_requires_grad(false);

      user_op::InputArgModifier* moving_max = GetInputArgModifierFn("moving_max", 0);
      CHECK(moving_max != nullptr);
      moving_max->set_requires_grad(false);
      moving_max->set_is_mutable(true);

      user_op::InputArgModifier* moving_min = GetInputArgModifierFn("moving_min", 0);
      CHECK(moving_min != nullptr);
      moving_min->set_requires_grad(false);
      moving_min->set_is_mutable(true);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // NOTE(Liang Depeng): all inputs need to be broadcast in order to accuratly calculate the
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

      int64_t stop_update_after_iters = op_conf.attr<int64_t>("stop_update_after_iters");
      CHECK_GT_OR_RETURN(stop_update_after_iters, 0);

      std::string quantization_formula = op_conf.attr<std::string>("quantization_formula");
      CHECK_OR_RETURN(quantization_formula == "google" || quantization_formula == "cambricon");
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
