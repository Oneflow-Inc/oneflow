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
#include "oneflow/user/ops/loss_op_util.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const auto& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const auto& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_EQ_OR_RETURN(prediction_desc.is_dynamic(), label_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(prediction_desc.shape(), label_desc.shape());
  CHECK_GE_OR_RETURN(ctx->Attr<float>("beta"), 0);

  JUST(CheckLossReductionAndInferOutputTenserDesc(ctx, "loss", prediction_desc.is_dynamic(),
                                                  prediction_desc.shape()));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_EQ_OR_RETURN(prediction_desc.data_type(), label_desc.data_type());

  *ctx->OutputDType("loss", 0) = ctx->InputDType("prediction", 0);

  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const Shape& prediction_shape = ctx->InputShape("prediction", 0);
  const Shape& label_shape = ctx->InputShape("label", 0);
  CHECK_EQ_OR_RETURN(prediction_shape, label_shape);
  JUST(CheckLossReductionAndCheckInputTenserDesc(ctx, "loss_grad", label_shape));

  CHECK_GE_OR_RETURN(ctx->Attr<float>("beta"), 0);

  *ctx->OutputShape("prediction_grad", 0) = prediction_shape;
  return Maybe<void>::Ok();
}

Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("loss_grad", 0), ctx->InputDType("prediction", 0));
  CHECK_EQ_OR_RETURN(ctx->InputDType("prediction", 0), ctx->InputDType("label", 0));
  *ctx->OutputDType("prediction_grad", 0) = ctx->InputDType("loss_grad", 0);
  return Maybe<void>::Ok();
}

}  // namespace
REGISTER_USER_OP("smooth_l1_loss")
    .Input("prediction")
    .Input("label")
    .Output("loss")
    .Attr<float>("beta")
    .Attr<std::string>("reduction")
    .SetTensorDescInferFn(InferTensorDescFn)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) -> Maybe<void> {
      user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);
      CHECK_OR_RETURN(label_modifier != nullptr);
      label_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn(InferDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& prediction_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
      FOR_RANGE(int64_t, i, 0, prediction_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("smooth_l1_loss_grad")
    .Input("loss_grad")
    .Input("prediction")
    .Input("label")
    .Output("prediction_grad")
    .Attr<float>("beta")
    .Attr<std::string>("reduction")
    .SetTensorDescInferFn(InferGradTensorDescFn)
    .SetDataTypeInferFn(InferGradDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& prediction_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
      FOR_RANGE(int64_t, i, 0, prediction_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("smooth_l1_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("prediction", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("smooth_l1_loss_grad")
                .Input("loss_grad", op.GetGradTensorWithOpOutput("loss", 0))
                .Input("prediction", op.input("prediction", 0))
                .Input("label", op.input("label", 0))
                .Output("prediction_grad")
                .Attr("beta", op.attr<float>("beta"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_grad", 0), "prediction", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
