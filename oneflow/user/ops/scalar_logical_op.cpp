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

#define REGISTER_SCALAR_LOGICAL_OP(op_name)                                           \
  REGISTER_NO_GRAD_USER_OP(op_name)                                                   \
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
        *ctx->OutputDType("out", 0) = DataType::kInt8;                                \
        return Maybe<void>::Ok();                                                     \
      });

REGISTER_SCALAR_LOGICAL_OP("scalar_logical_equal");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_not_equal");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_greater");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_greater_equal");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_less");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_less_equal");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_and");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_or");
REGISTER_SCALAR_LOGICAL_OP("scalar_logical_xor");
}  // namespace oneflow
