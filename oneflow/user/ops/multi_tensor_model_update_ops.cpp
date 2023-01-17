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
  CHECK_EQ_OR_RETURN(tensor_desc->shape(), like->shape())
      << "Tensordesc shape should be equal to Like shape. ";
  return Maybe<void>::Ok();
}

Maybe<void> CheckDataTypeLike(const user_op::TensorDesc* tensor_desc,
                              const user_op::TensorDesc* like) {
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), like->data_type())
      << "Tensordesc DataType should be equal to Like DataType. ";
  return Maybe<void>::Ok();
}

Maybe<void> CheckScalarShape(const user_op::TensorDesc* tensor_desc) {
  CHECK_OR_RETURN(tensor_desc->shape().NumAxes() == 0
                  || (tensor_desc->shape().NumAxes() == 1 && tensor_desc->shape().At(0) == 1))
      << tensor_desc->shape().DebugStr();
  return Maybe<void>::Ok();
}

Maybe<void> CheckScalarDataType(const user_op::TensorDesc* tensor_desc, const DataType data_type) {
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), data_type)
      << "TensorDesc DataType should be equal to Scalar DataType. ";
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

Maybe<void> SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL_OR_RETURN(arg_modifier) << "Arg Modifier should not be null. ";
  arg_modifier->set_is_mutable(true);
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model Diff shape should be equal to Model shape. ";
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
  const user_op::TensorDesc& first_model_desc = ctx->InputTensorDesc("model", 0);
  const int64_t input_size = ctx->input_size("model");
  for (int64_t i = 0; i < input_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    CHECK_EQ(model.data_type(), first_model_desc.data_type()) << "Model DataType should be equal. ";
  }
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, first_model_desc.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SgdInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferMomentumUpdateTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    const user_op::TensorDesc& momentum_buf = ctx->InputTensorDesc("momentum_buf", i);
    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model Diff shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(momentum_buf.shape(), model.shape())
        << "Momentum buf shape should be equal to Model shape. ";
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferMomentumUpdateDataType(user_op::InferContext* ctx) {
  JUST(CheckLearningRateDataType(ctx));
  const user_op::TensorDesc& first_model_desc = ctx->InputTensorDesc("model", 0);
  const int64_t input_size = ctx->input_size("model");
  for (int64_t i = 0; i < input_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& momentum_buf = ctx->InputTensorDesc("momentum_buf", i);
    CHECK_EQ(model.data_type(), first_model_desc.data_type()) << "Model DataType should be equal. ";
    CHECK_EQ(momentum_buf.data_type(), first_model_desc.data_type())
        << "Momentum buf DataType should be equal. ";
  }
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, first_model_desc.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> MomentumInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                     const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "momentum_buf", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferAdamUpdateTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    const user_op::TensorDesc& m = ctx->InputTensorDesc("m", i);
    const user_op::TensorDesc& v = ctx->InputTensorDesc("v", i);

    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model Diff shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(m.shape(), model.shape()) << "m shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(v.shape(), model.shape()) << "v shape should be equal to Model shape. ";
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferAdamUpdateDataType(user_op::InferContext* ctx) {  // todo
  JUST(CheckLearningRateDataType(ctx));
  const user_op::TensorDesc& first_model_desc = ctx->InputTensorDesc("model", 0);
  const int64_t input_size = ctx->input_size("model");
  for (int64_t i = 0; i < input_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& m = ctx->InputTensorDesc("m", i);
    const user_op::TensorDesc& v = ctx->InputTensorDesc("v", i);
    CHECK_EQ(model.data_type(), first_model_desc.data_type()) << "Model DataType should be equal. ";
    CHECK_EQ(m.data_type(), first_model_desc.data_type()) << "m DataType should be equal. ";
    CHECK_EQ(v.data_type(), first_model_desc.data_type()) << "v DataType should be equal. ";
  }
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, first_model_desc.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> AdamInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "m", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "v", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateWithCastTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_copy = ctx->InputTensorDesc("model_copy", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model diff shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(model_copy.shape(), model.shape())
        << "Model copy shape should be equal to Model shape. ";
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SgdWithCastInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                        const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_copy", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferMomentumUpdateWithCastTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_copy = ctx->InputTensorDesc("model_copy", i);
    const user_op::TensorDesc& momentum_buf = ctx->InputTensorDesc("momentum_buf", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model diff shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(momentum_buf.shape(), model.shape())
        << "Momentum buf shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(model_copy.shape(), model.shape())
        << "Model copy shape should be equal to Model shape. ";
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> MomentumWithCastInputArgModifyFn(
    const user_op::GetInputArgModifier& GetInputArgModifierFn,
    const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "momentum_buf", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_copy", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferAdamUpdateWithCastTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", i);
    const user_op::TensorDesc& model_copy = ctx->InputTensorDesc("model_copy", i);
    const user_op::TensorDesc& m = ctx->InputTensorDesc("m", i);
    const user_op::TensorDesc& v = ctx->InputTensorDesc("v", i);

    CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape())
        << "Model diff shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(model_copy.shape(), model.shape())
        << "Model copy shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(m.shape(), model.shape()) << "m shape should be equal to Model shape. ";
    CHECK_EQ_OR_RETURN(v.shape(), model.shape()) << "v shape should be equal to Model shape. ";
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> AdamWithCastInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                         const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_copy", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "m", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "v", i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferYoloV5WeightUpdateTensorDesc(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("model");
  for (int i = 0; i < weight_size; i++) {
    const user_op::TensorDesc& model_i = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_update_i = ctx->InputTensorDesc("model_update", i);
    CHECK_EQ_OR_RETURN(model_update_i.shape(), model_i.shape())
        << "All Model shape should be equal to model_update shape.";
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferYoloV5WeightUpdateDataType(user_op::InferContext* ctx) {
  JUST(CheckLearningRateDataType(ctx));
  const user_op::TensorDesc& first_model_desc = ctx->InputTensorDesc("model", 0);
  const int64_t input_size = ctx->input_size("model");
  for (int64_t i = 0; i < input_size; i++) {
    const user_op::TensorDesc& model = ctx->InputTensorDesc("model", i);
    const user_op::TensorDesc& model_update_i = ctx->InputTensorDesc("model_update", i);
    CHECK_EQ(model.data_type(), first_model_desc.data_type()) << "Model DataType should be equal. ";
    CHECK_EQ(model_update_i.data_type(), first_model_desc.data_type())
        << "Model DataType should be equal to model_update DataType.";
  }
  return Maybe<void>::Ok();
}

Maybe<void> YoloV5WeightInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                         const user_op::UserOpConfWrapper& conf) {
  for (int64_t i = 0; i < conf.input_size("model"); i++) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", i));
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_update", i));
  }
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
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorSgdUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return SgdInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorSgdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferSGDUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferMomentumUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorMomentumUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorMomentumUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return MomentumInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferMomentumUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorAdamUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferAdamUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorAdamUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorAdamUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorAdamUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdamInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorAdamUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferAdamUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorSgdUpdateWithCastOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferSGDUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorSgdUpdateWithCastOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorSgdUpdateWithCastOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorSgdUpdateWithCastOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return SgdWithCastInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorSgdUpdateWithCastOp::InferDataType(user_op::InferContext* ctx) {
  return InferSGDUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateWithCastOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferMomentumUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorMomentumUpdateWithCastOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateWithCastOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorMomentumUpdateWithCastOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return MomentumWithCastInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorMomentumUpdateWithCastOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferMomentumUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorAdamUpdateWithCastOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferAdamUpdateWithCastTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorAdamUpdateWithCastOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorAdamUpdateWithCastOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorAdamUpdateWithCastOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdamWithCastInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorAdamUpdateWithCastOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferAdamUpdateDataType(ctx);
}

/* static */ Maybe<void> MultiTensorYoloV5WeightUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferYoloV5WeightUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MultiTensorYoloV5WeightUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiTensorYoloV5WeightUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiTensorYoloV5WeightUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return YoloV5WeightInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MultiTensorYoloV5WeightUpdateOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferYoloV5WeightUpdateDataType(ctx);
}

}  // namespace oneflow
