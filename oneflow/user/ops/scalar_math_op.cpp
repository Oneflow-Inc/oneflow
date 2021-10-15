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

#define REGISTER_SCALAR_MATH_OP(op_name)                                              \
  REGISTER_USER_OP(op_name)                                                           \
      .Input("in")                                                                    \
      .Output("out")                                                                  \
      .Attr<bool>("has_int_operand")                                                  \
      .Attr<bool>("has_float_operand")                                                \
      .Attr<int64_t>("int_operand")                                                   \
      .Attr<double>("float_operand")                                                  \
      .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {           \
        *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);                       \
        *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);               \
        return Maybe<void>::Ok();                                                     \
      })                                                                              \
      .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {                      \
        const user_op::TensorDesc& in_tensor =                                        \
            ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);                     \
        FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {                       \
          ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build(); \
        }                                                                             \
        return Maybe<void>::Ok();                                                     \
      })                                                                              \
      .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {             \
        *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);                       \
        return Maybe<void>::Ok();                                                     \
      });

REGISTER_SCALAR_MATH_OP("scalar_add")
REGISTER_SCALAR_MATH_OP("scalar_floordiv")
REGISTER_SCALAR_MATH_OP("scalar_fmod")
REGISTER_SCALAR_MATH_OP("scalar_mul")
REGISTER_SCALAR_MATH_OP("scalar_pow")

REGISTER_USER_OP("scalar_pow_grad")
    .Input("x")
    .Input("dy")
    .Attr<bool>("has_int_operand")
    .Attr<bool>("has_float_operand")
    .Attr<int64_t>("int_operand")
    .Attr<double>("float_operand")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("dx", 0) = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), ctx->InputDType("dy", 0));
      *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("scalar_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "in", 0);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("scalar_mul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_mul")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr("has_int_operand", op.attr<bool>("has_int_operand"))
                .Attr("int_operand", op.attr_or_default<int64_t>("int_operand", 0))
                .Attr("has_float_operand", op.attr<bool>("has_float_operand"))
                .Attr("float_operand", op.attr_or_default<double>("float_operand", 0.0))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("scalar_pow")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_pow_grad")
                .Input("x", op.input("in", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("dx")
                .Attr("has_int_operand", op.attr<bool>("has_int_operand"))
                .Attr("int_operand", op.attr_or_default<int64_t>("int_operand", 0))
                .Attr("has_float_operand", op.attr<bool>("has_float_operand"))
                .Attr("float_operand", op.attr_or_default<double>("float_operand", 0.0))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "in", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
