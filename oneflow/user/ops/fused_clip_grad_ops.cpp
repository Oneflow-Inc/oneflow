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
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL_OR_RETURN(arg_modifier) << "Arg Modifier should not be null. ";
  arg_modifier->set_is_mutable(true);
  return Maybe<void>::Ok();
}

Maybe<void> InputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("grad"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "grad", i));
  }
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedClipGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedClipGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedClipGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedClipGradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return InputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> FusedClipGradOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

}  // namespace oneflow
