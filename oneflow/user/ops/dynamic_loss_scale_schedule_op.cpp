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

namespace {

bool IsScalarTensor(const user_op::TensorDesc* desc) {
  return desc->shape().NumAxes() == 1 && desc->shape().At(0) == 1;
}

bool IsTensorWithType(const user_op::TensorDesc* desc, DataType data_type) {
  return desc->data_type() == data_type;
}

}  // namespace

/* static */ Maybe<void> DynamicLossScaleScheduleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  CHECK_OR_RETURN(IsScalarTensor(&(ctx->InputTensorDesc("count_not_finite", 0))));
  CHECK_OR_RETURN(IsScalarTensor(&(ctx->InputTensorDesc("loss_scale", 0))));
  CHECK_OR_RETURN(IsScalarTensor(&(ctx->InputTensorDesc("good_step_counter", 0))));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DynamicLossScaleScheduleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DynamicLossScaleScheduleOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> DynamicLossScaleScheduleOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* loss_scale = GetInputArgModifierFn("loss_scale", 0);
  CHECK_OR_RETURN(loss_scale != nullptr);
  loss_scale->set_is_mutable(true);
  user_op::InputArgModifier* good_step_counter = GetInputArgModifierFn("good_step_counter", 0);
  CHECK_OR_RETURN(good_step_counter != nullptr);
  good_step_counter->set_is_mutable(true);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DynamicLossScaleScheduleOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(
      IsTensorWithType(&(ctx->InputTensorDesc("count_not_finite", 0)), DataType::kInt64));
  CHECK_OR_RETURN(IsTensorWithType(&(ctx->InputTensorDesc("loss_scale", 0)), DataType::kFloat));
  CHECK_OR_RETURN(
      IsTensorWithType(&(ctx->InputTensorDesc("good_step_counter", 0)), DataType::kInt64));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
