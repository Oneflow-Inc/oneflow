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
  CHECK_EQ_OR_RETURN(tensor_desc->shape(), Shape({1}));
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
  CHECK_EQ_OR_RETURN(model->data_type(), model_diff_values->data_type());
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
  const user_op::TensorDesc& beta1_t = ctx->InputTensorDesc("beta1_t", 0);
  const user_op::TensorDesc& beta2_t = ctx->InputTensorDesc("beta2_t", 0);
  JUST(CheckScalarShape(&beta1_t));
  JUST(CheckScalarShape(&beta2_t));
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
  const DataType data_type = model.data_type();
  const user_op::TensorDesc& beta1_t = ctx->InputTensorDesc("beta1_t", 0);
  const user_op::TensorDesc& beta2_t = ctx->InputTensorDesc("beta2_t", 0);
  JUST(CheckScalarDataType(&beta1_t, data_type));
  JUST(CheckScalarDataType(&beta2_t, data_type));
  JUST(CheckLearningRateDataType(ctx));
  if (ctx->has_input("scale_by_tensor", 0)) {
    const auto& scale_by_tensor = ctx->InputTensorDesc("scale_by_tensor", 0);
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

Maybe<void> AdamInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "m", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "v", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "max_v", 0));
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
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "beta1_t", 0));
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "beta2_t", 0));
  return Maybe<void>::Ok();
}

Maybe<void> SgdInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const user_op::UserOpConfWrapper& conf) {
  JUST(SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0));
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
REGISTER_NO_GRAD_USER_OP("sgd_update")
    .Input("model")
    .Input("model_diff")
    .OptionalInput("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferSGDUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(ctx->inputs())
            .Split(user_op::OpArg("model", 0), axis)
            .Split(user_op::OpArg("model_diff", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(SgdInputArgModifyFn)
    .SetDataTypeInferFn(InferSGDUpdateDataType);

REGISTER_NO_GRAD_USER_OP("indexed_slices_sgd_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferIndexedSlicesSGDUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(IndexedSlicesSgdInputArgModifyFn)
    .SetDataTypeInferFn(InferIndexedSlicesSGDUpdateDataType);

REGISTER_NO_GRAD_USER_OP("momentum_update")
    .Input("model")
    .Input("model_diff")
    .Input("momentum")
    .OptionalInput("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("beta", 0.9)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferMomentumUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetInputArgModifyFn(MomentumInputArgModifyFn)
    .SetDataTypeInferFn(InferMomentumUpdateDataType);

REGISTER_NO_GRAD_USER_OP("indexed_slices_momentum_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Input("momentum")
    .Attr<float>("beta", 0.9)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferIndexedSlicesMomentumUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Split(user_op::OpArg("momentum", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(IndexedSlicesMomentumInputArgModifyFn)
    .SetDataTypeInferFn(InferIndexedSlicesMomentumUpdateDataType);

REGISTER_NO_GRAD_USER_OP("adam_update")
    .Input("model")
    .Input("model_diff")
    .OptionalInput("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .OptionalInput("bias_correction1")
    .OptionalInput("bias_correction2")
    .Input("m")
    .Input("v")
    .Input("max_v")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<float>("bias_correction1_val", 1.0)
    .Attr<float>("bias_correction2_val", 1.0)
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("beta1", 0.9)
    .Attr<float>("beta2", 0.999)
    .Attr<float>("epsilon", 1e-8)
    .Attr<float>("weight_decay", 0.0)
    .Attr<bool>("amsgrad", false)
    .Attr<bool>("do_bias_correction", true)
    .SetTensorDescInferFn(InferAdamUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(ctx->inputs())
            .Split(user_op::OpArg("model", 0), axis)
            .Split(user_op::OpArg("model_diff", 0), axis)
            .Split(user_op::OpArg("m", 0), axis)
            .Split(user_op::OpArg("v", 0), axis)
            .Split(user_op::OpArg("max_v", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(AdamInputArgModifyFn)
    .SetDataTypeInferFn(InferAdamUpdateDataType);

REGISTER_NO_GRAD_USER_OP("adagrad_update")
    .Input("model")
    .Input("model_diff")
    .OptionalInput("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .OptionalInput("train_step")
    .Input("sum")
    .Attr<int>("train_step_val", 0)
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("lr_decay", 0.0)
    .Attr<float>("weight_decay", 0.0)
    .Attr<float>("epsilon", 1e-10)
    .SetTensorDescInferFn(InferAdagradUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetInputArgModifyFn(AdagradInputArgModifyFn)
    .SetDataTypeInferFn(InferAdagradUpdateDataType);

REGISTER_NO_GRAD_USER_OP("indexed_slices_adam_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .OptionalInput("bias_correction1")
    .OptionalInput("bias_correction2")
    .Input("m")
    .Input("v")
    .Input("max_v")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<float>("beta1", 0.9)
    .Attr<float>("beta2", 0.999)
    .Attr<float>("epsilon", 1e-8)
    .Attr<float>("weight_decay", 0.0)
    .Attr<bool>("amsgrad", false)
    .Attr<bool>("do_bias_correction", true)
    .SetTensorDescInferFn(InferIndexedSlicesAdamUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      const user_op::TensorDesc& model_diff_indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
      std::vector<user_op::OpArg> broadcast_args;
      broadcast_args.emplace_back("learning_rate", 0);
      broadcast_args.emplace_back("model_diff_indices", 0);
      ctx->NewBuilder()
          .Broadcast(broadcast_args)
          .Broadcast(user_op::OpArg("model_diff_values", 0))
          .Split(user_op::OpArg("model", 0), 0)
          .Split(user_op::OpArg("m", 0), 0)
          .Split(user_op::OpArg("v", 0), 0)
          .Split(user_op::OpArg("max_v", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(broadcast_args)
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Split(user_op::OpArg("m", 0), i)
            .Split(user_op::OpArg("v", 0), i)
            .Split(user_op::OpArg("max_v", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(AdamInputArgModifyFn)
    .SetDataTypeInferFn(InferIndexedSlicesAdamUpdateDataType);

REGISTER_NO_GRAD_USER_OP("lamb_update")
    .Input("m")
    .Input("v")
    .Input("beta1_t")
    .Input("beta2_t")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .Attr<float>("beta1")
    .Attr<float>("beta2")
    .Attr<float>("epsilon")
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferLambUpdateTensorDesc)
    // every bn has sbp broadcast signature
    .SetInputArgModifyFn(LambInputArgModifyFn)
    .SetDataTypeInferFn(InferLambUpdateDataType)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("adam_bias_correction_factor")
    .Input("train_step")
    .Output("out")
    .Attr<float>("beta", 0.9)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("train_step", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = DataType::kFloat;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

// every bn has sbp broadcast signature

REGISTER_NO_GRAD_USER_OP("rmsprop_update")
    .Input("model")
    .Input("model_diff")
    .OptionalInput("learning_rate")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .Input("mean_square")
    .OptionalInput("mean_gradient")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<bool>("centered", false)
    .Attr<float>("epsilon", 1e-8)
    .Attr<float>("decay_rate", 0.99)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferRmsPropUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetInputArgModifyFn(RmsPropUpdateInputArgModifyFn)
    .SetDataTypeInferFn(InferRmsPropUpdateDataType);

REGISTER_NO_GRAD_USER_OP("lars_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .Input("momentum")
    .OptionalInput("scale_by_tensor")
    .OptionalInput("skip_if")
    .Attr<double>("scale", 1.0)
    .Attr<float>("l1", 0.0)
    .Attr<float>("l2", 0.0)
    .Attr<float>("momentum_beta", 0.9)
    .Attr<float>("epsilon", 1e-9)
    .Attr<float>("lars_coefficient", 1e-4)
    .Attr<float>("weight_decay", 0.0)
    .SetTensorDescInferFn(InferLarsUpdateTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetInputArgModifyFn(LarsUpdateInputArgModifyFn)
    .SetDataTypeInferFn(InferLarsUpdateDataType);

}  // namespace

}  // namespace oneflow
