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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferFWTensorDesc(user_op::InferContext* ctx) {
  std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
  const Shape& x_shape = ctx->InputShape("x", 0);
  DimVector out_shape(x_shape.NumAxes());
  out_shape[0] = x_shape.dim_vec()[0];
  out_shape[1] = x_shape.dim_vec()[1];
  for (int i = 2; i < out_shape.size(); ++i) {
    out_shape[i] = output_size.size() > i - 2 ? output_size[i - 2] : output_size[0];
  }

  ctx->SetOutputShape("y", 0, Shape(out_shape));
  return Maybe<void>::Ok();
}

Maybe<void> InferBWTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  ctx->SetOutputIsDynamic("dx", 0, ctx->InputIsDynamic("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  // only for nchw
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferFWDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferBWDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace

#define DEF_ADAPTIVE_AVG_POOL_OP(op_class_name_prefix)                                           \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferLogicalTensorDesc(                     \
      user_op::InferContext* ctx) {                                                              \
    return InferFWTensorDesc(ctx);                                                               \
  }                                                                                              \
                                                                                                 \
  /*static*/ Maybe<void> op_class_name_prefix##Op::InferPhysicalTensorDesc(                      \
      user_op::InferContext* ctx) {                                                              \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::GetSbp(user_op::SbpContext* ctx) {          \
    return FwGetSbpFn(ctx);                                                                      \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferDataType(user_op::InferContext* ctx) { \
    return InferFWDataType(ctx);                                                                 \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::InferLogicalTensorDesc(                 \
      user_op::InferContext* ctx) {                                                              \
    return InferBWTensorDesc(ctx);                                                               \
  }                                                                                              \
                                                                                                 \
  /*static*/ Maybe<void> op_class_name_prefix##GradOp::InferPhysicalTensorDesc(                  \
      user_op::InferContext* ctx) {                                                              \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::GetSbp(user_op::SbpContext* ctx) {      \
    return BwGetSbpFn(ctx);                                                                      \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##GradOp::InferDataType(                          \
      user_op::InferContext* ctx) {                                                              \
    return InferBWDataType(ctx);                                                                 \
  }

DEF_ADAPTIVE_AVG_POOL_OP(AdaptiveAvgPool1D)
DEF_ADAPTIVE_AVG_POOL_OP(AdaptiveAvgPool2D)
DEF_ADAPTIVE_AVG_POOL_OP(AdaptiveAvgPool3D)

#undef DEF_ADAPTIVE_AVG_POOL_OP

}  // namespace oneflow
