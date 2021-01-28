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

namespace oneflow {



#define MATH_ELEMENTWISE_UNARY_SET_FUNC_NORMAL()                       \
  SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged) \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)       \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)

#define MATH_ELEMENTWISE_UNARY_SET_FUNC_LOGICAL()                       \
    SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {                \
      *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x",0);    \
      *ctx->Dtype4ArgNameAndIndex("y", 0) = DataType::kInt8;                            \
      return Maybe<void>::Ok();                                                          \
    })                                                                                  \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)       \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)

#define MATH_ELEMENTWISE_UNARY_SET_FUNC_BACKWARD_NORMAL()                       \
    SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {                \
      CHECK_OR_RETURN(*ctx->Shape4ArgNameAndIndex("x",0) == *ctx->Shape4ArgNameAndIndex("y",0)); \
      CHECK_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("y", 0) == DataType::kInt8);             \
      *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);  \
      return Maybe<void>::Ok();                                                          \
    })                                                                                  \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)       \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)

#define MATH_ELEMENTWISE_UNARY_SET_FUNC_BACKWARD_LOGICAL()                       \
  SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged) \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)       \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(math_unary_elementwise_type, func_prefix, tensor_suffix) \
  REGISTER_USER_OP(math_unary_elementwise_type)                                               \
      .Input("x")                                                                             \
      .Output("y")                                                                            \
      .MATH_ELEMENTWISE_UNARY_SET_FUNC_##tensor_suffix();                                                                   \
                                                                                              \
  REGISTER_USER_OP((std::string("") + math_unary_elementwise_type + "_grad"))                 \
      .Input("x")                                                                             \
      .Input("dy")                                                                            \
      .Output("dx")                                                                           \
     .MATH_ELEMENTWISE_UNARY_SET_FUNC_BACKWARD_##tensor_suffix();                                                                   \
                                                                                              \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                          \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {  \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                           \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");                  \
          user_op::UserOpConfWrapper unary_grad_op =                                          \
              builder.Op(std::string("") + math_unary_elementwise_type + "_grad")             \
                  .Input("x", op.input("x", 0))                                               \
                  .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                          \
                  .Output("dx")                                                               \
                  .Build();                                                                   \
          op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);                \
          AddOp(unary_grad_op);                                                               \
        }                                                                                     \
      });

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_NORMAL(op_name, fun_prefix) \
  REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(op_name, fun_prefix, NORMAL)
#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_LOGICAL(op_name, fun_prefix) \
  REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(op_name, fun_prefix, LOGICAL)

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_NORMAL, MATH_UNARY_ELEMENTWISE_FUNC_SEQ)
OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_LOGICAL,
MATH_UNARY_ELEMENTWISE_LOGICAL_FUNC_SEQ)

}  // namespace oneflow
