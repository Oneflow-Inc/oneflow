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
#include "oneflow/user/ops/math_binary_elementwise_seq.h"

namespace oneflow {

#define MATH_ELEMENTWISE_DEFAULT_SET_FUNC()                       \
  SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged) \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)

#define REGISTER_MATH_BINARY_ELEMENTWISE_OP_AND_GRAD(math_binary_elementwise_type, func_prefix) \
  REGISTER_USER_OP(math_binary_elementwise_type)                                                \
      .Input("x")                                                                               \
      .Input("y")                                                                               \
      .Output("z")                                                                              \
      .MATH_ELEMENTWISE_DEFAULT_SET_FUNC();                                                     \
                                                                                                \
  REGISTER_USER_OP((std::string("") + math_binary_elementwise_type + "_x_grad"))                \
      .Input("x")                                                                               \
      .Input("y")                                                                               \
      .Input("dz")                                                                              \
      .Output("dx")                                                                             \
      .MATH_ELEMENTWISE_DEFAULT_SET_FUNC();                                                     \
                                                                                                \
  REGISTER_USER_OP((std::string("") + math_binary_elementwise_type + "_y_grad"))                \
      .Input("x")                                                                               \
      .Input("y")                                                                               \
      .Input("dz")                                                                              \
      .Output("dy")                                                                             \
      .MATH_ELEMENTWISE_DEFAULT_SET_FUNC();                                                     \
                                                                                                \
  REGISTER_USER_OP_GRAD(math_binary_elementwise_type)                                           \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {    \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                             \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");                  \
          user_op::UserOpConfWrapper binary_grad_op =                                           \
              builder.Op(std::string("") + math_binary_elementwise_type + "_x_grad")            \
                  .Input("x", op.input("x", 0))                                                 \
                  .Input("y", op.input("y", 0))                                                 \
                  .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                            \
                  .Output("dx")                                                                 \
                  .Build();                                                                     \
          op.BindGradTensorWithOpInput(binary_grad_op.output("dx", 0), "x", 0);                 \
          AddOp(binary_grad_op);                                                                \
        }                                                                                       \
        if (op.NeedGenGradTensor4OpInput("y", 0)) {                                             \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");                  \
          user_op::UserOpConfWrapper binary_grad_op =                                           \
              builder.Op(std::string("") + math_binary_elementwise_type + "_y_grad")            \
                  .Input("x", op.input("x", 0))                                                 \
                  .Input("y", op.input("y", 0))                                                 \
                  .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                            \
                  .Output("dy")                                                                 \
                  .Build();                                                                     \
          op.BindGradTensorWithOpInput(binary_grad_op.output("dy", 0), "y", 0);                 \
          AddOp(binary_grad_op);                                                                \
        }                                                                                       \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_OP_AND_GRAD, MATH_BINARY_ELEMENTWISE_FUNC_SEQ)

}  // namespace oneflow
