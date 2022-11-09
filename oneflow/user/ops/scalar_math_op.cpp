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

namespace {

Maybe<void> GetSbp4ScalarMath(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbp4ScalarMul(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

}  // namespace

#define IMPLEMENT_SCALAR_MATH_OP_FUNCS(op_name, get_sbp_fn)                                        \
  /*static*/ Maybe<void> op_name##Op::GetSbp(user_op::SbpContext* ctx) { return get_sbp_fn(ctx); } \
  /*static*/ Maybe<void> op_name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {         \
    ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));                                       \
    ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));                               \
    return Maybe<void>::Ok();                                                                      \
  }                                                                                                \
  /*static*/ Maybe<void> op_name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {        \
    return InferLogicalTensorDesc(ctx);                                                            \
  }                                                                                                \
  /*static*/ Maybe<void> op_name##Op::InferDataType(user_op::InferContext* ctx) {                  \
    ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));                                       \
    return Maybe<void>::Ok();                                                                      \
  }

IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarAdd, GetSbp4ScalarMath)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarFloordiv, GetSbp4ScalarMath)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarTruncdiv, GetSbp4ScalarMath)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarFmod, GetSbp4ScalarMath)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarMul, GetSbp4ScalarMul)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarDiv, GetSbp4ScalarMul)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarPow, GetSbp4ScalarMath)
IMPLEMENT_SCALAR_MATH_OP_FUNCS(ScalarReversePow, GetSbp4ScalarMath)
#undef IMPLEMENT_SCALAR_MATH_OP_FUNCS

/*static*/ Maybe<void> ScalarPowGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ScalarPowGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ScalarPowGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarPowGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), ctx->InputDType("dy", 0))
      << Error::TypeError() << "Tensors dy and x must have same type";
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ScalarReversePowGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ScalarReversePowGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ScalarReversePowGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarReversePowGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), ctx->InputDType("dy", 0))
      << Error::TypeError() << "Tensors dy and x must have same type";
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
