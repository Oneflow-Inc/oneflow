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

REGISTER_USER_OP("softmax_cross_entropy")
    .Input("prediction")
    .Input("label")
    .Output("prob")  //'prob' is just for compute prediction's grad, prob's grad will be ignored
    .Output("out")
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prediction_desc = ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_EQ_OR_RETURN(prediction_desc->is_dynamic(), label_desc->is_dynamic());
      CHECK_GE_OR_RETURN(prediction_desc->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(label_desc->shape(), prediction_desc->shape());
      CHECK_EQ_OR_RETURN(label_desc->data_type(), prediction_desc->data_type());
      const int64_t num_out_axes = prediction_desc->shape().NumAxes() - 1;
      DimVector out_dim_vector;
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        out_dim_vector.push_back(prediction_desc->shape().At(i));
      }
      *ctx->TensorDesc4ArgNameAndIndex("prob", 0) = *prediction_desc;
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *prediction_desc;
      *out_desc->mut_shape() = Shape(out_dim_vector);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // ctx->LogicalTensorDesc4InputArgNameAndIndex("out", 0) is not initialized here
      const auto num_out_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0).shape().NumAxes() - 1;
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("prediction", 0), i)
            .Split(user_op::OpArg("label", 0), i)
            .Split(user_op::OpArg("prob", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("softmax_cross_entropy_grad")
    .Input("dy")
    .Input("label")
    .Input("prob")
    .Output("prediction_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prob_desc = ctx->TensorDesc4ArgNameAndIndex("prob", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      CHECK_EQ_OR_RETURN(prob_desc->is_dynamic(), label_desc->is_dynamic());
      CHECK_GE_OR_RETURN(prob_desc->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(dy_desc->shape().NumAxes(), prob_desc->shape().NumAxes() - 1);
      FOR_RANGE(int64_t, i, 0, dy_desc->shape().NumAxes()) {
        CHECK_EQ_OR_RETURN(dy_desc->shape().At(i), label_desc->shape().At(i));
      }
      CHECK_EQ_OR_RETURN(label_desc->shape(), prob_desc->shape());
      CHECK_EQ_OR_RETURN(label_desc->data_type(), prob_desc->data_type());
      CHECK_EQ_OR_RETURN(dy_desc->data_type(), prob_desc->data_type());
      *ctx->TensorDesc4ArgNameAndIndex("prediction_diff", 0) = *prob_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto num_dy_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0).shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, num_dy_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("dy", 0), i)
            .Split(user_op::OpArg("label", 0), i)
            .Split(user_op::OpArg("prob", 0), i)
            .Split(user_op::OpArg("prediction_diff", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("softmax_cross_entropy")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("prediction", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("softmax_cross_entropy_grad")
                .Input("prob", op.output("prob", 0))
                .Input("label", op.input("label", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("prediction_diff")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_diff", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
