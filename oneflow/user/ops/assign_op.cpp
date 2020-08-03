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

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* ref_desc = ctx->TensorDesc4ArgNameAndIndex("ref", 0);
  user_op::TensorDesc* value_desc = ctx->TensorDesc4ArgNameAndIndex("value", 0);
  CHECK_OR_RETURN(!ref_desc->is_dynamic());
  CHECK_OR_RETURN(ref_desc->shape() == value_desc->shape());
  CHECK_OR_RETURN(ref_desc->data_type() == value_desc->data_type());
  CHECK_OR_RETURN(ref_desc->is_tensor_list() == value_desc->is_tensor_list());
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0);
  FOR_RANGE(int64_t, axis, 0, ref_desc.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

void InputArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* ref_modifier = GetInputArgModifierFn("ref", 0);
  CHECK(ref_modifier != nullptr);
  ref_modifier->set_is_mutable(true);
  user_op::InputArgModifier* value_modifier = GetInputArgModifierFn("value", 0);
  CHECK(value_modifier != nullptr);
  value_modifier->set_requires_grad(false);
}

}  // namespace

REGISTER_USER_OP("assign")
    .Input("ref")
    .Input("value")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn(GetSbpSignatures)
    .SetInputArgModifyFn(InputArgModifierFn);

}  // namespace oneflow
