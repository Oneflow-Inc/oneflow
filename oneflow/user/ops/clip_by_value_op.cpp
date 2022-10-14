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

Maybe<void> InferClipTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("y", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetClipSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferClipGradTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetClipGradSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferClipTensorDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferClipGradDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace

#define DEF_CLIP_BY_VALUE_OP(op_class_name_prefix)                                               \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferLogicalTensorDesc(                     \
      user_op::InferContext* ctx) {                                                              \
    return InferClipTensorDesc(ctx);                                                             \
  }                                                                                              \
                                                                                                 \
  /*static*/ Maybe<void> op_class_name_prefix##Op::InferPhysicalTensorDesc(                      \
      user_op::InferContext* ctx) {                                                              \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::GetSbp(user_op::SbpContext* ctx) {          \
    return GetClipSbpSignature(ctx);                                                             \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferDataType(user_op::InferContext* ctx) { \
    return InferClipTensorDataType(ctx);                                                         \
  }                                                                                              \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::InferLogicalTensorDesc(                 \
      user_op::InferContext* ctx) {                                                              \
    return InferClipGradTensorDesc(ctx);                                                         \
  }                                                                                              \
  /*static*/ Maybe<void> op_class_name_prefix##GradOp::InferPhysicalTensorDesc(                  \
      user_op::InferContext* ctx) {                                                              \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::GetSbp(user_op::SbpContext* ctx) {      \
    return GetClipGradSbpSignature(ctx);                                                         \
  }                                                                                              \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::InferDataType(                          \
      user_op::InferContext* ctx) {                                                              \
    return InferClipGradDataType(ctx);                                                           \
  }

DEF_CLIP_BY_VALUE_OP(ClipByScalar)
DEF_CLIP_BY_VALUE_OP(ClipByScalarMin)
DEF_CLIP_BY_VALUE_OP(ClipByScalarMax)

#undef DEF_CLIP_BY_VALUE_OP

}  // namespace oneflow
