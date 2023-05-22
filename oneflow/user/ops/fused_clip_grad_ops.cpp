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
  for (int64_t i = 0; i < conf.input_size("model_diff"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_diff", i));
  }
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedClipGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("model_diff", 0);
  auto* out = ctx->MutOutputTensorDesc("out", 0);
  for (int64_t i = 1; i < ctx->input_size("model_diff"); ++i) {
    const auto& cur_in = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(in_0.shape(), cur_in.shape())
        << Error::RuntimeError()
        << "inconsistent tensor size, expected all tensor to have the same shape, "
        << "but got " << in_0.shape().DebugStr() << " and " << cur_in.shape().DebugStr();
  }
  out->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedClipGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedClipGradOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff", 0).shape().NumAxes();
  for (int64_t i = 0; i < num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedClipGradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return InputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> FusedClipGradOp::InferDataType(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("model_diff", 0);
  auto* out = ctx->MutOutputTensorDesc("out", 0);
  const DataType data_type = in_0.data_type();
  for (int64_t i = 1; i < ctx->input_size("model_diff"); ++i) {
    const auto& cur_in = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(cur_in.data_type(), data_type)
        << Error::RuntimeError() << ctx->op_name()
        << " expected all tenser to have same type, but found " << DataType_Name(cur_in.data_type())
        << " and " << DataType_Name(data_type);
  }
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
