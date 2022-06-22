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
  const int64_t weight_size = ctx->input_size("weights");
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  for (int idx = weight_size - 1; idx > -1; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weights", idx);
    *ctx->OutputShape("d_weights", idx) = weight_desc.shape();
    // this bias grad is previous layer, so here is weight_shape(1)
    if (idx != 0) { *ctx->OutputShape("d_biases", idx - 1) = Shape({weight_desc.shape().At(1)}); }
  }
  *ctx->OutputShape("d_grad", 0) = x_desc.shape();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBackward(user_op::InferContext* ctx) {
  const int64_t weight_size = ctx->input_size("weights");
  const int64_t dweight_size = ctx->output_size("d_weights");
  CHECK_EQ(weight_size, dweight_size) << "The number of weights and d_weights should be equal. "; 
  const int64_t dbias_size = ctx->output_size("d_biases");
  CHECK_EQ(weight_size - 1, dbias_size) << "The number of d_biases should be equal to weight_size - 1. Because last layer's bias_grad is computed by ReduceSum. "; 
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  for (int idx = weight_size - 1; idx > -1; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weights", idx);
    *ctx->OutputDType("d_weights", idx) = dy_desc.data_type();
    if (idx != 0) { *ctx->OutputDType("d_biases", idx - 1) = dy_desc.data_type(); }
  }
  *ctx->OutputDType("d_grad", 0) = dy_desc.data_type();
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
  for (int i = 0; i < ctx->user_op_conf().input_size("weights"); ++i) {
    builder.Broadcast(user_op::OpArg("weights", i));
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("cublas_aux"); ++i) {
    builder.Split(user_op::OpArg("cublas_aux", i), 0);
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("hidden"); ++i) {
    builder.Split(user_op::OpArg("hidden", i), 0);
  }

  builder.Split(user_op::OpArg("d_grad", 0), 0);
  for (int i = 0; i < ctx->user_op_conf().input_size("d_biases"); ++i) {
    builder.PartialSum(user_op::OpArg("d_biases", i));
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("d_weights"); ++i) {
    builder.PartialSum(user_op::OpArg("d_weights", i));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasFusedMLPGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBackward(ctx);
}

}  // namespace oneflow
