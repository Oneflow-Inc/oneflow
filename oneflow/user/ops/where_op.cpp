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

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  CHECK_EQ_OR_RETURN(*cond_shape, *ctx->Shape4ArgNameAndIndex("x", 0));
  CHECK_EQ_OR_RETURN(*cond_shape, *ctx->Shape4ArgNameAndIndex("y", 0));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *cond_shape;
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  CHECK_OR_RETURN(IsIntegralDataType(cond_dtype));
  DataType x_dtype = *ctx->Dtype4ArgNameAndIndex("x", 0);
  CHECK_EQ_OR_RETURN(x_dtype, *ctx->Dtype4ArgNameAndIndex("y", 0));
  *ctx->Dtype4ArgNameAndIndex("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& condition_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0);
  FOR_RANGE(int64_t, i, 0, condition_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("where")
    .Input("condition")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn(InferWhereTensorDesc)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("condition", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn(GetWhereSbpSignatures);

REGISTER_USER_OP_GRAD("where").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto zero_op_name = ctx->FwOp().op_name() + "_zero_grad";
  ctx->DefineOp(zero_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("zero_like")
        .InputBind("like", ctx->FwOp().input("x", 0))
        .Output("out")
        .Build();
  });

  const auto x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
  ctx->DefineOp(x_grad_op_name, [&ctx, &zero_op_name](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("where")
        .InputBind("condition", ctx->FwOp().input("condition", 0))
        .InputBind("x", ctx->FwOp().output_grad("out", 0))
        .InputBind("y", ctx->GetOp(zero_op_name).output("out", 0))
        .Output("out")
        .Build();
  });

  const auto y_grad_op_name = ctx->FwOp().op_name() + "_y_grad";
  ctx->DefineOp(y_grad_op_name, [&ctx, &zero_op_name](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("where")
        .InputBind("condition", ctx->FwOp().input("condition", 0))
        .InputBind("x", ctx->GetOp(zero_op_name).output("out", 0))
        .InputBind("y", ctx->FwOp().output_grad("out", 0))
        .Output("out")
        .Build();
  });

  ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                            [&ctx, &x_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(x_grad_op_name).output("out", 0);
                            });
  ctx->FwOp().InputGradBind(user_op::OpArg("y", 0),
                            [&ctx, &y_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(y_grad_op_name).output("out", 0);
                            });
});

}  // namespace oneflow
