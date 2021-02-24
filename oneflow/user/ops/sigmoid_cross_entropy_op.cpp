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

REGISTER_USER_OP("sigmoid_cross_entropy")
    .Input("prediction")
    .Input("label")
    .Output("loss")
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prediction_desc = ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_EQ_OR_RETURN(label_desc->shape(), prediction_desc->shape());
      user_op::TensorDesc* loss_desc = ctx->TensorDesc4ArgNameAndIndex("loss", 0);
      *loss_desc = *prediction_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto num_out_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0).shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("prediction", 0), i)
            .Split(user_op::OpArg("label", 0), i)
            .Split(user_op::OpArg("loss", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("sigmoid_cross_entropy_grad")
    .Input("prediction")
    .Input("loss_diff")
    .Input("label")
    .Output("prediction_diff")
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prediction_desc = ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      const user_op::TensorDesc* loss_diff_desc = ctx->TensorDesc4ArgNameAndIndex("loss_diff", 0);
      CHECK_EQ_OR_RETURN(label_desc->shape(), prediction_desc->shape());
      CHECK_EQ_OR_RETURN(loss_diff_desc->shape(), prediction_desc->shape());
      *ctx->TensorDesc4ArgNameAndIndex("prediction_diff", 0) = *prediction_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto num_dy_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("loss_diff", 0).shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, num_dy_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("loss_diff", 0), i)
            .Split(user_op::OpArg("label", 0), i)
            .Split(user_op::OpArg("prediction", 0), i)
            .Split(user_op::OpArg("prediction_diff", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("sigmoid_cross_entropy")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("prediction", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("sigmoid_cross_entropy_grad")
                .Input("prediction", op.input("prediction", 0))
                .Input("label", op.input("label", 0))
                .Input("loss_diff", op.GetGradTensorWithOpOutput("loss", 0))
                .Output("prediction_diff")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_diff", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });
}  // namespace oneflow
