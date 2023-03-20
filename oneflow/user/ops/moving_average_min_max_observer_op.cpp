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

/* static */ Maybe<void> MovingAverageMinMaxObserverOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& moving_max_shape = ctx->InputShape("moving_max", 0);
  const Shape& moving_min_shape = ctx->InputShape("moving_min", 0);
  const Shape& current_train_step = ctx->InputShape("current_train_step", 0);

  // NOTE(Liang Depeng): for now only support per-layer quantization
  // TODO(Liang Depeng): depthwise convolution support per-channel quantization
  CHECK_OR_RETURN(moving_max_shape.NumAxes() == 1 && moving_max_shape.At(0) == 1);
  CHECK_OR_RETURN(moving_min_shape.NumAxes() == 1 && moving_min_shape.At(0) == 1);

  CHECK_OR_RETURN(current_train_step.NumAxes() == 1 && current_train_step.At(0) == 1);

  ctx->SetOutputShape("scale", 0, Shape({1}));
  ctx->SetOutputShape("zero_point", 0, Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MovingAverageMinMaxObserverOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MovingAverageMinMaxObserverOp::GetSbp(user_op::SbpContext* ctx) {
  // NOTE(Liang Depeng): all inputs need to be broadcast in order to accuratly calculate the
  // global scale and zero_point
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MovingAverageMinMaxObserverOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
  CHECK_OR_RETURN(in != nullptr);
  in->set_requires_grad(false);

  user_op::InputArgModifier* current_train_step = GetInputArgModifierFn("current_train_step", 0);
  CHECK_OR_RETURN(current_train_step != nullptr);
  current_train_step->set_requires_grad(false);

  user_op::InputArgModifier* moving_max = GetInputArgModifierFn("moving_max", 0);
  CHECK_OR_RETURN(moving_max != nullptr);
  moving_max->set_requires_grad(false);
  moving_max->set_is_mutable(true);

  user_op::InputArgModifier* moving_min = GetInputArgModifierFn("moving_min", 0);
  CHECK_OR_RETURN(moving_min != nullptr);
  moving_min->set_requires_grad(false);
  moving_min->set_is_mutable(true);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MovingAverageMinMaxObserverOp::CheckAttr(
    const user_op::UserOpDefWrapper& def, const user_op::UserOpConfWrapper& op_conf) {
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
}

/* static */ Maybe<void> MovingAverageMinMaxObserverOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("scale", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("zero_point", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
