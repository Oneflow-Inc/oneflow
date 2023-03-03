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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType query_type = ctx->InputDType("x", 0);
  DataType mask_bias_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_bias_type, query_type);

  if (ctx->has_input("bias", 0)) {
    DataType bias_type = ctx->InputDType("bias", 0);
    CHECK_EQ_OR_RETURN(bias_type, query_type);
  }
  ctx->SetOutputDType("out", 0, query_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const float scale = ctx->Attr<float>("scale");
  CHECK_LE_OR_RETURN(scale, 1.);

  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  CHECK_OR_RETURN(x_shape[-1] == mask_shape[-1] && x_shape[0] == mask_shape[0]);
  if (ctx->has_input("bias", 0)) {
    const Shape& bias_shape = ctx->InputShape("bias", 0);
    CHECK_OR_RETURN(mask_shape[-1] == bias_shape[-1]);
    CHECK_OR_RETURN(mask_shape[0] == bias_shape[0] || bias_shape[0] == 1);
    for (int i = 1; i < x_shape.NumAxes() - 1; i++) {
      CHECK_OR_RETURN((mask_shape[i] == 1 || bias_shape[i] == 1)
                      && mask_shape[i] * bias_shape[i] == x_shape[i]);
    }
  } else {
    auto axes = x_shape.NumAxes();
    bool reach1 = false;
    for (int i = 0; i < axes - 1; i++) {
      CHECK_OR_RETURN((mask_shape[i] == x_shape[i] && !reach1) || (1 == mask_shape[i]));
      reach1 = (1 == mask_shape[i]);
    }
  }
  ctx->SetOutputShape("out", 0, x_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  if (ctx->Attr<bool>("inplace") == false)
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  else
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Broadcast(user_op::OpArg("bias", 0))
        .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType y_type = ctx->InputDType("y", 0);
  DataType dy_type = ctx->InputDType("dy", 0);
  CHECK_EQ_OR_RETURN(y_type, dy_type);
  ctx->SetOutputDType("dx", 0, y_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(y_shape, dy_shape);
  ctx->SetOutputShape("dx", 0, y_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
