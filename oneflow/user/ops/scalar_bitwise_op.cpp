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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

#define IMPLEMENT_SCALAR_BITWISE_OP_FUNCS(name)                                                  \
  /*static*/ Maybe<void> name##Op::GetSbp(user_op::SbpContext* ctx) {                            \
    const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0); \
    FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {                                      \
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();                \
    }                                                                                            \
    return Maybe<void>::Ok();                                                                    \
  }                                                                                              \
  /*static*/ Maybe<void> name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {          \
    ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));                                     \
    ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));                             \
    return Maybe<void>::Ok();                                                                    \
  }                                                                                              \
  /*static*/ Maybe<void> name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {         \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
  /*static*/ Maybe<void> name##Op::InferDataType(user_op::InferContext* ctx) {                   \
    ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));                                     \
    return Maybe<void>::Ok();                                                                    \
  }

IMPLEMENT_SCALAR_BITWISE_OP_FUNCS(ScalarBitwiseAnd);
IMPLEMENT_SCALAR_BITWISE_OP_FUNCS(ScalarBitwiseOr);
IMPLEMENT_SCALAR_BITWISE_OP_FUNCS(ScalarBitwiseXor);

}  // namespace oneflow
