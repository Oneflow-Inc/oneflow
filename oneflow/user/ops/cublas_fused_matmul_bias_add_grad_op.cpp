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

Maybe<void> InferTensorDesc4MatmulBiasAddBackward(user_op::InferContext* ctx) {
  /*
  x (m, k)
  w (n, k) need transpose
  bias (n, )
  y (m, n)
  w_grad = dy_transpose matmul x
  */
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);

  const int64_t bias_size = dy_desc.shape().At(1);
  Shape w_grad_shape({dy_desc.shape().At(1), x_desc.shape().At(1)});
  ctx->SetOutputShape("w_grad", 0, w_grad_shape);
  ctx->SetOutputShape("b_grad", 0, Shape({bias_size}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBiasAddBackward(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(x_desc.data_type(), dy_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(dy_desc.data_type()) << ", but got "
      << DataType_Name(x_desc.data_type());

  user_op::TensorDesc* w_grad_desc = ctx->MutOutputTensorDesc("w_grad", 0);
  user_op::TensorDesc* b_grad_desc = ctx->MutOutputTensorDesc("b_grad", 0);

  w_grad_desc->set_data_type(dy_desc.data_type());
  b_grad_desc->set_data_type(dy_desc.data_type());
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CublasMatmulBiasAddGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4MatmulBiasAddBackward(ctx);
}

/*static*/ Maybe<void> CublasMatmulBiasAddGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CublasMatmulBiasAddGradOp::GetSbp(user_op::SbpContext* ctx) {
  /*
  dy need transpose.

  assume dy(m, n), x(m, k), dbias=(n, 1)
  dw = dy_T matmul x

  */
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 1)
      .Broadcast(user_op::OpArg("x", 0))
      .Split(user_op::OpArg("w_grad", 0), 0)
      .Split(user_op::OpArg("b_grad", 0), 0)
      .Build();

  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .PartialSum(user_op::OpArg("w_grad", 0))
      .PartialSum(user_op::OpArg("b_grad", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasMatmulBiasAddGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBiasAddBackward(ctx);
}

}  // namespace oneflow
