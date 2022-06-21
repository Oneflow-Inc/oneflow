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
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4MatmulAsyncBackward(user_op::InferContext* ctx) {
  /*
  x (m, k)
  w (n, k) need transpose
  dy (m, n)
  d_weight = dy_transpose matmul x
  d_grad = dy matmul w
  */
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);

  CHECK_EQ_OR_RETURN(x_desc.shape().At(0), dy_desc.shape().At(0))
      << "M dim in x and dy should be equal";
  CHECK_EQ_OR_RETURN(x_desc.shape().At(1), weight_desc.shape().At(1))
      << "K dim in x and weight should be equal";
  CHECK_EQ_OR_RETURN(weight_desc.shape().At(0), dy_desc.shape().At(1))
      << "N dim in weight and dy should be equal";

  Shape d_weight_shape({dy_desc.shape().At(1), x_desc.shape().At(1)});
  Shape d_grad_shape({dy_desc.shape().At(0), weight_desc.shape().At(1)});

  *ctx->OutputShape("d_grad", 0) = d_grad_shape;
  *ctx->OutputShape("d_weight", 0) = d_weight_shape;
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulAsyncBackward(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(x_desc.data_type(), dy_desc.data_type())
      << "x's datatype should be the same as y's datatype";
  CHECK_EQ_OR_RETURN(x_desc.data_type(), weight_desc.data_type())
      << "x's datatype should be the same as weight's datatype";

  user_op::TensorDesc* d_grad_desc = ctx->OutputTensorDesc("d_grad", 0);
  user_op::TensorDesc* w_grad_desc = ctx->OutputTensorDesc("d_weight", 0);

  *d_grad_desc->mut_data_type() = dy_desc.data_type();
  *w_grad_desc->mut_data_type() = dy_desc.data_type();
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> MatmulAsyncGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4MatmulAsyncBackward(ctx);
}

/*static*/ Maybe<void> MatmulAsyncGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatmulAsyncGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("d_grad", 0), 0)
      .PartialSum(user_op::OpArg("d_weight", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MatmulAsyncGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulAsyncBackward(ctx);
}

}  // namespace oneflow
