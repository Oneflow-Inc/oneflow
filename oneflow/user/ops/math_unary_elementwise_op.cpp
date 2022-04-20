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
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

#define MATH_ELEMENTWISE_DEFAULT_SET_FUNC(op_type)                                       \
  /* static */ Maybe<void> op_type::InferLogicalTensorDesc(user_op::InferContext* ctx) { \
    return user_op::TensorDescInferFnUtil::Unchanged(ctx);                               \
  }                                                                                      \
  /*static*/ Maybe<void> op_type::InferPhysicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /* static */ Maybe<void> op_type::GetSbp(user_op::SbpContext* ctx) {                   \
    return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);                                 \
  }                                                                                      \
  /* static */ Maybe<void> op_type::InferDataType(user_op::InferContext* ctx) {          \
    return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);                       \
  }

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(math_unary_elementwise_type, func_prefix)  \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##Op)                                           \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##GradOp)                                       \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                           \
      .SetGenBackwardOpConfFn(                                                                 \
          [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> { \
            if (op.NeedGenGradTensor4OpInput("x", 0)) {                                        \
              user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");               \
              user_op::UserOpConfWrapper unary_grad_op =                                       \
                  builder.Op(std::string("") + math_unary_elementwise_type + "_grad")          \
                      .Input("x", op.input("x", 0))                                            \
                      .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                       \
                      .Output("dx")                                                            \
                      .Build();                                                                \
              op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);             \
              AddOp(unary_grad_op);                                                            \
            }                                                                                  \
            return Maybe<void>::Ok();                                                          \
          });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD,
                     MATH_UNARY_ELEMENTWISE_FUNC_SEQ_ODS)

}  // namespace oneflow
