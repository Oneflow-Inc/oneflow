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
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape());

  JUST(CheckLossReductionAndInferOutputTenserDesc(ctx, "out", input_desc.is_dynamic(),
                                                  input_desc.shape()));

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type());

  *ctx->OutputDType("out", 0) = ctx->InputDType("input", 0);

  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_EQ_OR_RETURN(input_desc.shape(), target_desc.shape());
  JUST(CheckLossReductionAndCheckInputTenserDesc(ctx, "dy", target_desc.shape()));

  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *dx_desc->mut_shape() = input_desc.shape();

  return Maybe<void>::Ok();
}

Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_EQ_OR_RETURN(input_desc.data_type(), target_desc.data_type());

  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("kl_div_loss")
    .Input("input")
    .Input("target")
    .Output("out")
    .Attr<std::string>("reduction")
    .Attr<bool>("log_target")
    .SetTensorDescInferFn(InferTensorDescFn)
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) -> Maybe<void> {
      user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
      CHECK_OR_RETURN(target_modifier != nullptr);
      target_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn(InferDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
      const auto reduction = ctx->Attr<std::string>("reduction");
      FOR_RANGE(int64_t, i, 0, input_shape.NumAxes()) {
        auto builder = ctx->NewBuilder().Split(ctx->inputs(), i);
        if (reduction != "none") {
          builder.Broadcast(user_op::OpArg("out", 0));
        } else {
          builder.Split(user_op::OpArg("out", 0), i);
        }
        builder.Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("kl_div_loss_grad")
    .Input("input")
    .Input("target")
    .Input("dy")
    .Output("dx")
    .Attr<std::string>("reduction")
    .Attr<bool>("log_target")
    .SetTensorDescInferFn(InferGradTensorDescFn)
    .SetDataTypeInferFn(InferGradDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
      const auto& dy_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0).shape();
      FOR_RANGE(int64_t, i, 0, input_shape.NumAxes()) {
        auto builder = ctx->NewBuilder()
                           .Split(user_op::OpArg("input", 0), i)
                           .Split(user_op::OpArg("target", 0), i)
                           .Split(user_op::OpArg("dx", 0), i);
        if (dy_shape.NumAxes() == 0) {
          builder.Broadcast(user_op::OpArg("dy", 0));
        } else {
          builder.Split(user_op::OpArg("dy", 0), i);
        }
        builder.Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("kl_div_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("kl_div_loss_grad")
                .Input("input", op.input("input", 0))
                .Input("target", op.input("target", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("dx")
                .Attr("reduction", op.attr<std::string>("reduction"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "input", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
