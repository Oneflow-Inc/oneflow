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
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), like->data_type())
      << "InferDataType Failed. Expected " << DataType_Name(tensor_desc->data_type())
      << ", but got " << DataType_Name(like->data_type());
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
      << "InferDataType Failed. Expected " << DataType_Name(tensor_desc->data_type())
      << ", but got " << DataType_Name(data_type);
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

Maybe<void> CheckIndexedSlicesModelDiffDesc(const user_op::TensorDesc* model,
                                            const user_op::TensorDesc* model_diff_indices,
                                            const user_op::TensorDesc* model_diff_values) {
  const int64_t num_indices_axes = model_diff_indices->shape().NumAxes();
  const int64_t num_values_axes = model_diff_values->shape().NumAxes();
  CHECK_GE_OR_RETURN(num_values_axes, num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ_OR_RETURN(model_diff_values->shape().At(i), model_diff_indices->shape().At(i));
  }
  const int64_t num_model_axes = model->shape().NumAxes();
  CHECK_EQ_OR_RETURN(num_model_axes, num_values_axes - num_indices_axes + 1);
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    CHECK_EQ_OR_RETURN(model->shape().At(i),
                       model_diff_values->shape().At(num_indices_axes + i - 1));
  }
  return Maybe<void>::Ok();
}
Maybe<void> CheckIndexedSlicesModelDiffDataType(const user_op::TensorDesc* model,
                                                const user_op::TensorDesc* model_diff_indices,
                                                const user_op::TensorDesc* model_diff_values) {
  CHECK_OR_RETURN(IsIndexDataType(model_diff_indices->data_type()));
  CHECK_EQ_OR_RETURN(model->data_type(), model_diff_values->data_type())
      << "InferDataType Failed. Expected " << DataType_Name(model->data_type()) << ", but got "
      << DataType_Name(model_diff_values->data_type());
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  if (shape.NumAxes() > 0 && model_diff.shape().NumAxes() > 0) {
    CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  }
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("model_copy", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputTensorDesc("model_copy", 0).shape(), shape)
        << "Model copy shape should be equal to Model shape. ";
  }
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

Maybe<void> InferIndexedSlicesSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(&model, &model_diff_indices, &model_diff_values));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}
Maybe<void> InferIndexedSlicesSGDUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDataType(&model, &model_diff_indices, &model_diff_values));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferMomentumUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), model.shape());
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckShapeLike(&momentum, &model));
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}
Maybe<void> InferMomentumUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckDataTypeLike(&momentum, &model));
  JUST(CheckLearningRateDataType(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, model.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferIndexedSlicesMomentumUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(&model, &model_diff_indices, &model_diff_values));
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckShapeLike(&momentum, &model));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}
Maybe<void> InferIndexedSlicesMomentumUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDataType(&model, &model_diff_indices, &model_diff_values));
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckDataTypeLike(&momentum, &model));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferAdamUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  const user_op::TensorDesc& m = ctx->InputTensorDesc("m", 0);
  JUST(CheckShapeLike(&m, &model));
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);
  JUST(CheckShapeLike(&v, &model));
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("model_copy", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputTensorDesc("model_copy", 0).shape(), shape)
        << "Model copy shape should be equal to Model shape. ";
  }
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}
Maybe<void> InferAdamUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& m = ctx->InputTensorDesc("m", 0);
  JUST(CheckDataTypeLike(&m, &model));
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);
  JUST(CheckDataTypeLike(&v, &model));
  JUST(CheckLearningRateDataType(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, model.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferAdagradUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  const user_op::TensorDesc& sum = ctx->InputTensorDesc("sum", 0);
  JUST(CheckShapeLike(&sum, &model));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferAdagradUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& sum = ctx->InputTensorDesc("sum", 0);
  JUST(CheckDataTypeLike(&sum, &model));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferIndexedSlicesAdamUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(&model, &model_diff_indices, &model_diff_values));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}
Maybe<void> InferIndexedSlicesAdamUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff_indices = ctx->InputTensorDesc("model_diff_indices", 0);
  const user_op::TensorDesc& model_diff_values = ctx->InputTensorDesc("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDataType(&model, &model_diff_indices, &model_diff_values));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferLambUpdateTensorDesc(user_op::InferContext* ctx) {
  const float beta1 = ctx->Attr<float>("beta1");
  const float beta2 = ctx->Attr<float>("beta2");
  CHECK_GE_OR_RETURN(beta1, 0);
  CHECK_LT_OR_RETURN(beta1, 1);
  CHECK_GE_OR_RETURN(beta2, 0);
  CHECK_LT_OR_RETURN(beta2, 1);
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);

  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  const user_op::TensorDesc& m = ctx->InputTensorDesc("m", 0);
  JUST(CheckShapeLike(&m, &model));
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);
  JUST(CheckShapeLike(&v, &model));
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferLambUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& m = ctx->InputTensorDesc("m", 0);
  JUST(CheckDataTypeLike(&m, &model));
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);
  JUST(CheckDataTypeLike(&v, &model));
  JUST(CheckLearningRateDataType(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, model.data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferFtrlUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape)
      << "Model Diff shape is not consistent with Weight shape. ";
  const user_op::TensorDesc& accumulate = ctx->InputTensorDesc("accumulate", 0);
  const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);
  JUST(CheckShapeLike(&accumulate, &model));
  JUST(CheckShapeLike(&z, &model));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferFtrlUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& accumulate = ctx->InputTensorDesc("accumulate", 0);
  const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);
  JUST(CheckDataTypeLike(&accumulate, &model));
  JUST(CheckDataTypeLike(&z, &model));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferAdadeltaUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  const user_op::TensorDesc& square_avgs = ctx->InputTensorDesc("square_avgs", 0);
  const user_op::TensorDesc& acc_deltas = ctx->InputTensorDesc("acc_deltas", 0);
  JUST(CheckShapeLike(&model_diff, &model));
  JUST(CheckShapeLike(&square_avgs, &model));
  JUST(CheckShapeLike(&acc_deltas, &model));
  JUST(CheckLearningRateShape(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InferAdadeltaUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& square_avgs = ctx->InputTensorDesc("square_avgs", 0);
  const user_op::TensorDesc& acc_deltas = ctx->InputTensorDesc("acc_deltas", 0);
  JUST(CheckDataTypeLike(&square_avgs, &model));
  JUST(CheckDataTypeLike(&acc_deltas, &model));
  JUST(CheckLearningRateDataType(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL_OR_RETURN(arg_modifier);
  arg_modifier->set_is_mutable(true);
  return Maybe<void>::Ok();
}

Maybe<void> AdamInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "m", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "v", 0));
  if (conf.has_input("max_v", 0)) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "max_v", 0));
  }
  if (conf.has_input("model_copy", 0)) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_copy", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> AdagradInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                    const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "sum", 0));
  return Maybe<void>::Ok();
}

Maybe<void> LambInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "m", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "v", 0));
  return Maybe<void>::Ok();
}

Maybe<void> SgdInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  if (conf.has_input("model_copy", 0)) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model_copy", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesSgdInputArgModifyFn(
    const user_op::GetInputArgModifier& GetInputArgModifierFn,
    const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  return Maybe<void>::Ok();
}

Maybe<void> MomentumInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                     const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "momentum", 0));
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesMomentumInputArgModifyFn(
    const user_op::GetInputArgModifier& GetInputArgModifierFn,
    const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "momentum", 0));
  return Maybe<void>::Ok();
}

Maybe<void> RmsPropUpdateInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                          const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "mean_square", 0));
  if (conf.attr<bool>("centered")) {
    JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "mean_gradient", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LarsUpdateInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                       const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "momentum", 0));
  return Maybe<void>::Ok();
}

Maybe<void> FtrlInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "accumulate", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "z", 0));
  return Maybe<void>::Ok();
}

Maybe<void> AdadeltaInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                     const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "square_avgs", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "acc_deltas", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferRmsPropUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);

  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  const user_op::TensorDesc& mean_square = ctx->InputTensorDesc("mean_square", 0);
  JUST(CheckShapeLike(&mean_square, &model));
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  if (ctx->Attr<bool>("centered")) {
    CHECK_OR_RETURN(ctx->has_input("mean_gradient", 0));
    const user_op::TensorDesc& mean_gradient = ctx->InputTensorDesc("mean_gradient", 0);
    JUST(CheckShapeLike(&mean_gradient, &model));
  } else {
    CHECK_OR_RETURN(!ctx->has_input("mean_gradient", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferRmsPropUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& mean_square = ctx->InputTensorDesc("mean_square", 0);
  JUST(CheckDataTypeLike(&mean_square, &model));
  JUST(CheckLearningRateDataType(ctx));
  const DataType data_type = model.data_type();
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, data_type));
  }
  if (ctx->Attr<bool>("centered")) {
    CHECK_OR_RETURN(ctx->has_input("mean_gradient", 0));
    const user_op::TensorDesc& mean_gradient = ctx->InputTensorDesc("mean_gradient", 0);
    JUST(CheckDataTypeLike(&mean_gradient, &model));
  }
  return Maybe<void>::Ok();
}
Maybe<void> InferLarsUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);

  const Shape& shape = model.shape();
  const user_op::TensorDesc& model_diff = ctx->InputTensorDesc("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff.shape(), shape);
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckShapeLike(&momentum, &model));
  JUST(CheckLearningRateShape(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarShape(&scale_by_tensor));
  }
  return Maybe<void>::Ok();
}
Maybe<void> InferLarsUpdateDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& model = ctx->InputTensorDesc("model", 0);
  const user_op::TensorDesc& momentum = ctx->InputTensorDesc("momentum", 0);
  JUST(CheckDataTypeLike(&momentum, &model));
  JUST(CheckLearningRateDataType(ctx));
  const DataType data_type = model.data_type();
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
    JUST(CheckScalarDataType(&scale_by_tensor, data_type));
  }
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> SgdUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferSGDUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> SgdUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SgdUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    auto builder = ctx->NewBuilder()
                       .Broadcast(ctx->inputs())
                       .Split(user_op::OpArg("model", 0), axis)
                       .Split(user_op::OpArg("model_diff", 0), axis);
    if (ctx->user_op_conf().has_input("model_copy", 0)) {
      builder.Split(user_op::OpArg("model_copy", 0), axis);
    }
    builder.Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SgdUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return SgdInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> SgdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferSGDUpdateDataType(ctx);
}

/* static */ Maybe<void> IndexedSlicesSgdUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferIndexedSlicesSGDUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> IndexedSlicesSgdUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> IndexedSlicesSgdUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  const user_op::TensorDesc& model_diff_indices =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("learning_rate", 0))
      .Broadcast(user_op::OpArg("model_diff_indices", 0))
      .Broadcast(user_op::OpArg("model_diff_values", 0))
      .Split(user_op::OpArg("model", 0), 0)
      .Build();
  FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("learning_rate", 0))
        .Broadcast(user_op::OpArg("model_diff_indices", 0))
        .Split(user_op::OpArg("model_diff_values", 0), model_diff_indices.shape().NumAxes() + i - 1)
        .Split(user_op::OpArg("model", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IndexedSlicesSgdUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return IndexedSlicesSgdInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> IndexedSlicesSgdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferIndexedSlicesSGDUpdateDataType(ctx);
}

/* static */ Maybe<void> MomentumUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferMomentumUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> MomentumUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MomentumUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(ctx->inputs())
        .Split(user_op::OpArg("model", 0), axis)
        .Split(user_op::OpArg("model_diff", 0), axis)
        .Split(user_op::OpArg("momentum", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MomentumUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return MomentumInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MomentumUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferMomentumUpdateDataType(ctx);
}

/* static */ Maybe<void> IndexedSlicesMomentumUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferIndexedSlicesMomentumUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> IndexedSlicesMomentumUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> IndexedSlicesMomentumUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  const user_op::TensorDesc& model_diff_indices =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("learning_rate", 0))
      .Broadcast(user_op::OpArg("model_diff_indices", 0))
      .Broadcast(user_op::OpArg("model_diff_values", 0))
      .Split(user_op::OpArg("model", 0), 0)
      .Split(user_op::OpArg("momentum", 0), 0)
      .Build();
  FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("learning_rate", 0))
        .Broadcast(user_op::OpArg("model_diff_indices", 0))
        .Split(user_op::OpArg("model_diff_values", 0), model_diff_indices.shape().NumAxes() + i - 1)
        .Split(user_op::OpArg("model", 0), i)
        .Split(user_op::OpArg("momentum", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IndexedSlicesMomentumUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return IndexedSlicesMomentumInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> IndexedSlicesMomentumUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferIndexedSlicesMomentumUpdateDataType(ctx);
}

/* static */ Maybe<void> AdamUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferAdamUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> AdamUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdamUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    std::vector<user_op::OpArg> split_args;
    split_args.emplace_back("model", 0);
    split_args.emplace_back("model_diff", 0);
    split_args.emplace_back("m", 0);
    split_args.emplace_back("v", 0);
    if (ctx->user_op_conf().has_input("max_v", 0)) { split_args.emplace_back("max_v", 0); }
    auto builder = ctx->NewBuilder().Broadcast(ctx->inputs()).Split(split_args, axis);
    if (ctx->user_op_conf().has_input("model_copy", 0)) {
      builder.Split(user_op::OpArg("model_copy", 0), axis);
    }
    builder.Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdamUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdamInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> AdamUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferAdamUpdateDataType(ctx);
}

/* static */ Maybe<void> AdagradUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferAdagradUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> AdagradUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdagradUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(ctx->inputs())
        .Split(user_op::OpArg("model", 0), axis)
        .Split(user_op::OpArg("model_diff", 0), axis)
        .Split(user_op::OpArg("sum", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdagradUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdagradInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> AdagradUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferAdagradUpdateDataType(ctx);
}

/* static */ Maybe<void> IndexedSlicesAdamUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferIndexedSlicesAdamUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> IndexedSlicesAdamUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> IndexedSlicesAdamUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  const user_op::TensorDesc& model_diff_indices =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
  std::vector<user_op::OpArg> broadcast_args;
  broadcast_args.emplace_back("learning_rate", 0);
  broadcast_args.emplace_back("model_diff_indices", 0);

  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("model", 0);
  split_args.emplace_back("m", 0);
  split_args.emplace_back("v", 0);
  if (ctx->user_op_conf().has_input("max_v", 0)) { split_args.emplace_back("max_v", 0); }

  ctx->NewBuilder()
      .Broadcast(broadcast_args)
      .Broadcast(user_op::OpArg("model_diff_values", 0))
      .Split(split_args, 0)
      .Build();

  FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(broadcast_args)
        .Split(user_op::OpArg("model_diff_values", 0), model_diff_indices.shape().NumAxes() + i - 1)
        .Split(split_args, i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IndexedSlicesAdamUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdamInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> IndexedSlicesAdamUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferIndexedSlicesAdamUpdateDataType(ctx);
}

/* static */ Maybe<void> LambUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferLambUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> LambUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LambUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> LambUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return LambInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> LambUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferLambUpdateDataType(ctx);
}

/* static */ Maybe<void> AdamBiasCorrectionFactorOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("train_step", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AdamBiasCorrectionFactorOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdamBiasCorrectionFactorOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> AdamBiasCorrectionFactorOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kFloat);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmspropUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferRmsPropUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> RmspropUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RmspropUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  bool centered = ctx->Attr<bool>("centered");
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    if (centered) {
      ctx->NewBuilder()
          .Broadcast(ctx->inputs())
          .Split(user_op::OpArg("model", 0), axis)
          .Split(user_op::OpArg("model_diff", 0), axis)
          .Split(user_op::OpArg("mean_square", 0), axis)
          .Split(user_op::OpArg("mean_gradient", 0), axis)
          .Build();
    } else {
      ctx->NewBuilder()
          .Broadcast(ctx->inputs())
          .Split(user_op::OpArg("model", 0), axis)
          .Split(user_op::OpArg("model_diff", 0), axis)
          .Split(user_op::OpArg("mean_square", 0), axis)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmspropUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return RmsPropUpdateInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> RmspropUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferRmsPropUpdateDataType(ctx);
}

/* static */ Maybe<void> LarsUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferLarsUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> LarsUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LarsUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(ctx->inputs())
        .Split(user_op::OpArg("model", 0), axis)
        .Split(user_op::OpArg("model_diff", 0), axis)
        .Split(user_op::OpArg("momentum", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LarsUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return LarsUpdateInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> LarsUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferLarsUpdateDataType(ctx);
}

/* static */ Maybe<void> FtrlUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return FtrlInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> FtrlUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferFtrlUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> FtrlUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FtrlUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(ctx->inputs())
        .Split(user_op::OpArg("model", 0), axis)
        .Split(user_op::OpArg("model_diff", 0), axis)
        .Split(user_op::OpArg("accumulate", 0), axis)
        .Split(user_op::OpArg("z", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FtrlUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferFtrlUpdateDataType(ctx);
}

/* static */ Maybe<void> AdadeltaUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return AdadeltaInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> AdadeltaUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferAdadeltaUpdateTensorDesc(ctx);
}

/*static*/ Maybe<void> AdadeltaUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdadeltaUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
  FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
    ctx->NewBuilder()
        .Broadcast(ctx->inputs())
        .Split(user_op::OpArg("model", 0), axis)
        .Split(user_op::OpArg("model_diff", 0), axis)
        .Split(user_op::OpArg("square_avgs", 0), axis)
        .Split(user_op::OpArg("acc_deltas", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdadeltaUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferAdadeltaUpdateDataType(ctx);
}

}  // namespace oneflow
