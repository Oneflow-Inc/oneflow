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

Maybe<void> InferTensorDesc4FusedMatmulBackward(user_op::InferContext* ctx) {
  /*
  x(m, k) matmul w1(n, k)_transpose

        matmul_out(m, n) + bias(n)

                hidden(m, n) matmul w2(k2, n)_transpose

                    out(m, k2) + bias2(k2)

                            hidden2(m, k2)
  */
  const int64_t weight_size = ctx->input_size("weight");
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  printf("x shape is: %ld, %ld \n", x_desc.shape().At(0), x_desc.shape().At(1));
  printf("Dy shape is: %ld, %ld \n", dy_desc.shape().At(0), dy_desc.shape().At(1));
  printf("weight size is: %ld \n", weight_size);
  for (int idx = weight_size - 1; idx > -1; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", idx);
    printf("weight shape is: %ld, %ld \n", weight_desc.shape().At(0), weight_desc.shape().At(1));

    if (idx != 0) { printf("D bias(previous layer) shape is: %ld\n", weight_desc.shape().At(1)); }
  }

  for (int idx = weight_size - 1; idx > -1; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", idx);
    *ctx->OutputShape("d_weight", idx) = weight_desc.shape();
    // this bias grad is previous layer, so here is wshape(1)
    if (idx != 0) { *ctx->OutputShape("d_bias", idx - 1) = Shape({weight_desc.shape().At(1)}); }
  }
  *ctx->OutputShape("d_grad", 0) = x_desc.shape();
  printf("Success === \n");
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBackward(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("weight");
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  for (int idx = weight_size - 1; idx > -1; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", idx);
    *ctx->OutputDType("d_weight", idx) = dy_desc.data_type();
    // this bias grad is previous layer, so here is wshape(1)
    if (idx != 0) { *ctx->OutputDType("d_bias", idx - 1) = dy_desc.data_type(); }
  }
  *ctx->OutputDType("d_grad", 0) = dy_desc.data_type();
  printf("Succes2222s === \n");

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CublasFusedMLPGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBackward(ctx);
}

/*static*/ Maybe<void> CublasFusedMLPGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CublasFusedMLPGradOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0);
  builder.Split(user_op::OpArg("dy", 0), 0);
  for (int i = 0; i < ctx->user_op_conf().input_size("weight"); ++i) {
    builder.Broadcast(user_op::OpArg("weight", i));
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("aux"); ++i) {
    builder.Split(user_op::OpArg("aux", i), 0);
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("hidden"); ++i) {
    builder.Split(user_op::OpArg("hidden", i), 0);
  }

  builder.Split(user_op::OpArg("d_grad", 0), 0);
  for (int i = 0; i < ctx->user_op_conf().input_size("d_bias"); ++i) {
    builder.PartialSum(user_op::OpArg("d_bias", i));
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("d_weight"); ++i) {
    builder.PartialSum(user_op::OpArg("d_weight", i));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasFusedMLPGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBackward(ctx);
}

}  // namespace oneflow
