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

Maybe<void> CheckModelDiffDesc(const user_op::TensorDesc* model,
                               const user_op::TensorDesc* model_diff_indices,
                               const user_op::TensorDesc* model_diff_values) {
  CHECK_OR_RETURN(IsIndexDataType(model_diff_indices->data_type()));
  const int64_t num_indices_axes = model_diff_indices->shape().NumAxes();
  const int64_t num_values_axes = model_diff_values->shape().NumAxes();
  CHECK_GE_OR_RETURN(num_values_axes, num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ_OR_RETURN(model_diff_values->shape().At(i), model_diff_indices->shape().At(i));
  }
  CHECK_EQ_OR_RETURN(model->data_type(), model_diff_values->data_type());
  const int64_t num_model_axes = model->shape().NumAxes();
  CHECK_EQ_OR_RETURN(num_model_axes, num_values_axes - num_indices_axes + 1);
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    CHECK_EQ_OR_RETURN(model->shape().At(i),
                       model_diff_values->shape().At(num_indices_axes + i - 1));
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("indexed_slices_sgd_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const user_op::TensorDesc* model_diff_indices =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
      const user_op::TensorDesc* model_diff_values =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
      CheckModelDiffDesc(model, model_diff_indices, model_diff_values);
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
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
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      user_op::InputArgModifier* model_modifier = GetInputArgModifierFn("model", 0);
      CHECK_NOTNULL(model_modifier);
      model_modifier->set_is_mutable(true);
    });

REGISTER_USER_OP("indexed_slices_momentum_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Input("momentum")
    .Attr<float>("beta", UserOpAttrType::kAtFloat, 0.9)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const user_op::TensorDesc* model_diff_indices =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
      const user_op::TensorDesc* model_diff_values =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
      CheckModelDiffDesc(model, model_diff_indices, model_diff_values);
      const user_op::TensorDesc* momentum = ctx->TensorDesc4ArgNameAndIndex("momentum", 0);
      CHECK_EQ_OR_RETURN(momentum->data_type(), model->data_type());
      CHECK_EQ_OR_RETURN(momentum->shape(), model->shape());
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
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
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      user_op::InputArgModifier* model_modifier = GetInputArgModifierFn("model", 0);
      CHECK_NOTNULL(model_modifier);
      model_modifier->set_is_mutable(true);
      user_op::InputArgModifier* momentum_modifier = GetInputArgModifierFn("momentum", 0);
      CHECK_NOTNULL(momentum_modifier);
      momentum_modifier->set_is_mutable(true);
    });

REGISTER_USER_OP("indexed_slices_adam_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Input("m")
    .Input("v")
    .OptionalInput("beta1_t")
    .OptionalInput("beta2_t")
    .Attr<float>("beta1", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("beta2", UserOpAttrType::kAtFloat, 0.999)
    .Attr<float>("epsilon", UserOpAttrType::kAtFloat, 1e-8)
    .Attr<bool>("do_bias_correction", UserOpAttrType::kAtBool, false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const user_op::TensorDesc* model_diff_indices =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
      const user_op::TensorDesc* model_diff_values =
          ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
      CheckModelDiffDesc(model, model_diff_indices, model_diff_values);
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);

      if (ctx->Attr<bool>("do_bias_correction")) {
        CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta1_t", 0));
        CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta2_t", 0));
        const user_op::TensorDesc* beta1_t = ctx->TensorDesc4ArgNameAndIndex("beta1_t", 0);
        CHECK_EQ_OR_RETURN(beta1_t->shape(), Shape({1}));
        const user_op::TensorDesc* beta2_t = ctx->TensorDesc4ArgNameAndIndex("beta2_t", 0);
        CHECK_EQ_OR_RETURN(beta2_t->shape(), Shape({1}));
        CHECK_EQ_OR_RETURN(beta1_t->data_type(), beta2_t->data_type());
      } else {
        CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta1_t", 0));
        CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta2_t", 0));
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      const user_op::TensorDesc& model_diff_indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
      std::vector<user_op::OpArg> broadcast_args;
      broadcast_args.emplace_back("learning_rate", 0);
      broadcast_args.emplace_back("model_diff_indices", 0);
      if (ctx->Attr<bool>("do_bias_correction")) {
        broadcast_args.emplace_back("beta1_t", 0);
        broadcast_args.emplace_back("beta2_t", 0);
      }
      ctx->NewBuilder()
          .Broadcast(broadcast_args)
          .Broadcast(user_op::OpArg("model_diff_values", 0))
          .Split(user_op::OpArg("model", 0), 0)
          .Split(user_op::OpArg("m", 0), 0)
          .Split(user_op::OpArg("v", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(broadcast_args)
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Split(user_op::OpArg("m", 0), i)
            .Split(user_op::OpArg("v", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      user_op::InputArgModifier* model_modifier = GetInputArgModifierFn("model", 0);
      CHECK_NOTNULL(model_modifier);
      model_modifier->set_is_mutable(true);
      user_op::InputArgModifier* m_modifier = GetInputArgModifierFn("m", 0);
      CHECK_NOTNULL(m_modifier);
      m_modifier->set_is_mutable(true);
      user_op::InputArgModifier* v_modifier = GetInputArgModifierFn("v", 0);
      CHECK_NOTNULL(v_modifier);
      v_modifier->set_is_mutable(true);
      if (conf.attr<bool>("do_bias_correction")) {
        user_op::InputArgModifier* beta1_t_modifier = GetInputArgModifierFn("beta1_t", 0);
        CHECK_NOTNULL(beta1_t_modifier);
        beta1_t_modifier->set_is_mutable(true);
        user_op::InputArgModifier* beta2_t_modifier = GetInputArgModifierFn("beta2_t", 0);
        CHECK_NOTNULL(beta2_t_modifier);
        beta2_t_modifier->set_is_mutable(true);
      }
    });

}  // namespace

}  // namespace oneflow
