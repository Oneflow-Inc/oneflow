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

bool IsScalarTensorWithType(const user_op::TensorDesc* desc, DataType data_type) {
  return desc->shape().NumAxes() == 1 && desc->shape().At(0) == 1 && desc->data_type() == data_type;
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(IsScalarTensorWithType(ctx->TensorDesc4ArgNameAndIndex("count_not_finite", 0),
                                         DataType::kInt64));
  CHECK_OR_RETURN(
      IsScalarTensorWithType(ctx->TensorDesc4ArgNameAndIndex("loss_scale", 0), DataType::kFloat));
  CHECK_OR_RETURN(IsScalarTensorWithType(ctx->TensorDesc4ArgNameAndIndex("good_step_counter", 0),
                                         DataType::kInt64));
  return Maybe<void>::Ok();
}

void InputArgModifierFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* loss_scale = GetInputArgModifierFn("loss_scale", 0);
  CHECK(loss_scale != nullptr);
  loss_scale->set_is_mutable(true);
  user_op::InputArgModifier* good_step_counter = GetInputArgModifierFn("good_step_counter", 0);
  CHECK(good_step_counter != nullptr);
  good_step_counter->set_is_mutable(true);
}

}  // namespace

REGISTER_USER_OP("dynamic_loss_scale_schedule")
    .Input("count_not_finite")
    .Input("loss_scale")
    .Input("good_step_counter")
    .Attr<int64_t>("increment_period", 2000)
    .Attr<float>("multiplier", 2.0)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn(InputArgModifierFn);

}  // namespace oneflow
