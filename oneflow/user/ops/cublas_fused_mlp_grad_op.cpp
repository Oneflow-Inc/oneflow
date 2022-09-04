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

Maybe<void> InferTensorDesc4FusedMatmulBackward(user_op::InferContext* ctx) {
  const int64_t weight_num = ctx->input_size("weights");
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  for (int idx = weight_num - 1; idx >= 0; idx--) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weights", idx);
    ctx->SetOutputShape("d_weights", idx, weight_desc.shape());
    ctx->SetOutputShape("d_biases", idx, Shape({weight_desc.shape().At(0)}));
  }
  ctx->SetOutputShape("d_x", 0, x_desc.shape());
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBackward(user_op::InferContext* ctx) {
  const int64_t weight_num = ctx->input_size("weights");
  const int64_t dweight_num = ctx->output_size("d_weights");
  CHECK_EQ(weight_num, dweight_num) << "The number of weights and d_weights should be equal. ";
  const int64_t dbias_size = ctx->output_size("d_biases");
  CHECK_EQ(weight_num, dbias_size) << "The number of d_biases should be equal to weight_num. "
                                      "Because last layer's bias_grad is computed by ReduceSum. ";
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  for (int idx = weight_num - 1; idx >= 0; idx--) {
    ctx->SetOutputDType("d_weights", idx, dy_desc.data_type());
    ctx->SetOutputDType("d_biases", idx, dy_desc.data_type());
  }
  ctx->SetOutputDType("d_x", 0, dy_desc.data_type());
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

  builder.Split(user_op::OpArg("d_x", 0), 0);
  if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSED_MLP_GRAD_OVERLAP_ALLREDUCE", false)) {
    // FusedMLPGradKernel do allreduce for dbias and dweight, so here convert from PartialSum to
    // Broadcast.
    for (int i = 0; i < ctx->user_op_conf().output_size("d_biases"); ++i) {
      builder.Broadcast(user_op::OpArg("d_biases", i));
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("d_weights"); ++i) {
      builder.Broadcast(user_op::OpArg("d_weights", i));
    }
  } else {
    for (int i = 0; i < ctx->user_op_conf().output_size("d_biases"); ++i) {
      builder.PartialSum(user_op::OpArg("d_biases", i));
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("d_weights"); ++i) {
      builder.PartialSum(user_op::OpArg("d_weights", i));
    }
  }

  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasFusedMLPGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBackward(ctx);
}

}  // namespace oneflow
