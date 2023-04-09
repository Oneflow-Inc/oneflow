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

Maybe<void> SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL_OR_RETURN(arg_modifier);
  arg_modifier->set_is_mutable(true);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPUpdateScaleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AMPUpdateScaleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AMPUpdateScaleOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPUpdateScaleOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPUpdateScaleOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "current_scale", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "growth_tracker", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPForEachNonFiniteCheckAndUnscaleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AMPForEachNonFiniteCheckAndUnscaleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AMPForEachNonFiniteCheckAndUnscaleOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPForEachNonFiniteCheckAndUnscaleOp::InferDataType(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AMPForEachNonFiniteCheckAndUnscaleOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("scaled_grads_found_inf_inv_scale"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "scaled_grads_found_inf_inv_scale", i));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorAMPForEachNonFiniteCheckAndUnscaleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MultiTensorAMPForEachNonFiniteCheckAndUnscaleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorAMPForEachNonFiniteCheckAndUnscaleOp::GetSbp(
    user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorAMPForEachNonFiniteCheckAndUnscaleOp::InferDataType(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorAMPForEachNonFiniteCheckAndUnscaleOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("scaled_grads_found_inf_inv_scale"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "scaled_grads_found_inf_inv_scale", i));
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
