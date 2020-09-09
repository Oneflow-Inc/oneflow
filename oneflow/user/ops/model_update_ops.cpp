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

REGISTER_USER_OP("sgd_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .Attr<float>("scale", UserOpAttrType::kAtFloat, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const Shape& shape = model->shape();
      const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
      CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(ctx->inputs(), axis)
            .Broadcast(user_op::OpArg("learning_rate", 0))
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

REGISTER_USER_OP("momentum_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .Input("momentum")
    .Attr<float>("scale", UserOpAttrType::kAtFloat, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("beta", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const DataType data_type = model->data_type();
      const Shape& shape = model->shape();
      const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
      CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
      const user_op::TensorDesc* momentum = ctx->TensorDesc4ArgNameAndIndex("momentum", 0);
      CHECK_EQ_OR_RETURN(momentum->data_type(), data_type);
      CHECK_EQ_OR_RETURN(momentum->shape(), shape);
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(ctx->inputs(), axis)
            .Broadcast(user_op::OpArg("learning_rate", 0))
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

REGISTER_USER_OP("adam_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .Input("m")
    .Input("v")
    .OptionalInput("beta1_t")
    .OptionalInput("beta2_t")
    .Attr<float>("scale", UserOpAttrType::kAtFloat, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("beta1", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("beta2", UserOpAttrType::kAtFloat, 0.999)
    .Attr<float>("epsilon", UserOpAttrType::kAtFloat, 1e-8)
    .Attr<bool>("do_bias_correction", UserOpAttrType::kAtBool, false)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
      const DataType data_type = model->data_type();
      const Shape& shape = model->shape();
      const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
      CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
      const user_op::TensorDesc* m = ctx->TensorDesc4ArgNameAndIndex("m", 0);
      CHECK_EQ_OR_RETURN(m->data_type(), data_type);
      CHECK_EQ_OR_RETURN(m->shape(), shape);
      const user_op::TensorDesc* v = ctx->TensorDesc4ArgNameAndIndex("v", 0);
      CHECK_EQ_OR_RETURN(v->data_type(), data_type);
      CHECK_EQ_OR_RETURN(v->shape(), shape);
      const user_op::TensorDesc* learning_rate =
          ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
      CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
      CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
      if (ctx->Attr<bool>("do_bias_correction")) {
        CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta1_t", 0));
        CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta2_t", 0));
        const user_op::TensorDesc* beta1_t = ctx->TensorDesc4ArgNameAndIndex("beta1_t", 0);
        CHECK_EQ_OR_RETURN(beta1_t->shape(), Shape({1}));
        CHECK_EQ_OR_RETURN(beta1_t->data_type(), data_type);
        const user_op::TensorDesc* beta2_t = ctx->TensorDesc4ArgNameAndIndex("beta2_t", 0);
        CHECK_EQ_OR_RETURN(beta2_t->shape(), Shape({1}));
        CHECK_EQ_OR_RETURN(beta2_t->data_type(), data_type);
      } else {
        CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta1_t", 0));
        CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta2_t", 0));
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(ctx->inputs())
            .Split(user_op::OpArg("model", 0), axis)
            .Split(user_op::OpArg("model_diff", 0), axis)
            .Split(user_op::OpArg("m", 0), axis)
            .Split(user_op::OpArg("v", 0), axis)
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
