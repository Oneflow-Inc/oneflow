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

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const user_op::TensorDesc& value_desc = ctx->InputTensorDesc("value", 0);
  CHECK_OR_RETURN(!ref_desc.is_dynamic());
  CHECK_OR_RETURN(ref_desc.shape() == value_desc.shape());
  if (ctx->has_input("condition", 0)) {
    const user_op::TensorDesc& condition = ctx->InputTensorDesc("condition", 0);
    CHECK_OR_RETURN(condition.shape().NumAxes() == 1);
    CHECK_OR_RETURN(condition.shape().At(0) == 1);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0);
  FOR_RANGE(int64_t, axis, 0, ref_desc.shape().NumAxes()) {
    if (ctx->user_op_conf().has_input("condition", 0)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("ref", 0), axis)
          .Split(user_op::OpArg("value", 0), axis)
          .Broadcast(user_op::OpArg("condition", 0))
          .Build();
    } else {
      ctx->NewBuilder().Split(ctx->inputs(), axis).Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InputArgModifierFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                               const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* ref_modifier = GetInputArgModifierFn("ref", 0);
  CHECK_OR_RETURN(ref_modifier != nullptr);
  ref_modifier->set_is_mutable(true);
  user_op::InputArgModifier* value_modifier = GetInputArgModifierFn("value", 0);
  CHECK_OR_RETURN(value_modifier != nullptr);
  value_modifier->set_requires_grad(false);
  if (conf.has_input("condition", 0)) {
    user_op::InputArgModifier* condition_modifier = GetInputArgModifierFn("condition", 0);
    CHECK_OR_RETURN(condition_modifier != nullptr);
    condition_modifier->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType_(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const user_op::TensorDesc& value_desc = ctx->InputTensorDesc("value", 0);
  CHECK_OR_RETURN(ref_desc.data_type() == value_desc.data_type())
      << Error::RuntimeError() << DataType_Name(ref_desc.data_type()) << " vs."
      << DataType_Name(value_desc.data_type());
  if (ctx->has_input("condition", 0)) {
    const user_op::TensorDesc& condition = ctx->InputTensorDesc("condition", 0);
    CHECK_OR_RETURN(IsIndexDataType(condition.data_type()));
  }
  return Maybe<void>::Ok();
}

}  // namespace

#define DEF_ASSIGN_OP(op_class_name)                                                              \
  /* static */ Maybe<void> op_class_name::InferLogicalTensorDesc(user_op::InferContext* ctx) {    \
    return InferTensorDesc(ctx);                                                                  \
  }                                                                                               \
                                                                                                  \
  /*static*/ Maybe<void> op_class_name::InferPhysicalTensorDesc(user_op::InferContext* ctx) {     \
    return InferLogicalTensorDesc(ctx);                                                           \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::GetSbp(user_op::SbpContext* ctx) {                      \
    return GetSbpSignatures(ctx);                                                                 \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::ModifyInputArg(                                         \
      const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) { \
    return InputArgModifierFn(GetInputArgModifierFn, conf);                                       \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::InferDataType(user_op::InferContext* ctx) {             \
    return InferDataType_(ctx);                                                                   \
  }

DEF_ASSIGN_OP(AssignUserOp)
DEF_ASSIGN_OP(AssignIfOp)
DEF_ASSIGN_OP(AssignIfNotOp)

#undef DEF_ASSIGN_OP

}  // namespace oneflow
