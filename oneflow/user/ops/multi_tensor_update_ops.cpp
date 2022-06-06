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

Maybe<void> CheckShapeLike(const user_op::TensorDesc* tensor_desc,
                           const user_op::TensorDesc* like) {
  CHECK_EQ_OR_RETURN(tensor_desc->shape(), like->shape());
  return Maybe<void>::Ok();
}

Maybe<void> CheckDataTypeLike(const user_op::TensorDesc* tensor_desc,
                              const user_op::TensorDesc* like) {
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), like->data_type());
  return Maybe<void>::Ok();
}

Maybe<void> CheckScalarShape(const user_op::TensorDesc* tensor_desc) {
  CHECK_OR_RETURN(tensor_desc->shape().NumAxes() == 0
                  || (tensor_desc->shape().NumAxes() == 1 && tensor_desc->shape().At(0) == 1))
      << tensor_desc->shape().DebugStr();
  return Maybe<void>::Ok();
}

Maybe<void> CheckScalarDataType(const user_op::TensorDesc* tensor_desc, const DataType data_type) {
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), data_type);
  return Maybe<void>::Ok();
}

Maybe<void> CheckLearningRateShape(user_op::InferContext* ctx) {
  if (ctx->has_input("learning_rate", 0)) {
    const user_op::TensorDesc& learning_rate = ctx->InputTensorDesc("learning_rate", 0);
    JUST(CheckScalarShape(&learning_rate));
  }
  return Maybe<void>::Ok();
}
Maybe<void> CheckLearningRateDataType(user_op::InferContext* ctx) {
  if (ctx->has_input("learning_rate", 0)) {
    const user_op::TensorDesc& learning_rate = ctx->InputTensorDesc("learning_rate", 0);
    JUST(CheckScalarDataType(&learning_rate, DataType::kFloat));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape());
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateDataType(user_op::InferContext* ctx) {
  JUST(CheckLearningRateDataType(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, model.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL_OR_RETURN(arg_modifier);
  arg_modifier->set_is_mutable(true);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> MultiTensorSgdUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferSGDUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorSgdUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorSgdUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    auto builder = ctx->NewBuilder().Broadcast(ctx->inputs());
    for (int i = 0; i < ctx->user_op_conf().input_size("model"); ++i) {
      builder.Split(user_op::OpArg("model", i), axis);
      builder.Split(user_op::OpArg("model_diff", i), axis);
    }
    builder.Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> SgdInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorSgdUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return SgdInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorSgdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferSGDUpdateDataType(ctx);
}

}  // namespace oneflow
