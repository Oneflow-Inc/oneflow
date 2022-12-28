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

constexpr int32_t kAuxReluLdAlignRequirement = 128;

long AlignReluAuxLd(long aux_ld) {
  /*
  ReLu bit-mask matrix leading dimension in elements.
  Must be divisible by 128 and be no less than the number of rows in the output matrix.
  */
  long old_aux_ld = aux_ld;
  return ((old_aux_ld + kAuxReluLdAlignRequirement - 1) / kAuxReluLdAlignRequirement)
         * kAuxReluLdAlignRequirement;
}

Maybe<void> InferTensorDesc4FusedMatmul(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  int32_t weight_size = ctx->input_size("weights");
  int32_t bias_size = ctx->input_size("biases");
  CHECK_EQ_OR_RETURN(weight_size, bias_size) << "Weight num should be equal to bias num. ";
  /*
  A: (m, k)
  B: (n, k) need transpose
  C: (m, n)
  */
  int64_t m = 0, n = 0, k = 0, cublas_aux_ld = 0;
  m = x_desc.shape().At(0);
  k = x_desc.shape().At(1);

  for (int32_t idx = 0; idx < weight_size; idx++) {
    // skip first input weight.
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weights", idx);
    const user_op::TensorDesc& bias_desc = ctx->InputTensorDesc("biases", idx);
    CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2) << "Weight's ndim should be equal to 2. ";
    CHECK_EQ_OR_RETURN(bias_desc.shape().NumAxes(), 1) << "Bias's ndim should be equal to 1. ";

    n = weight_desc.shape().At(0);
    CHECK_EQ_OR_RETURN(bias_desc.shape().At(0), n)
        << "Bias shape should be equal to N. Assume (M, K) matmul (N, K, transpose_b=True) "
           "bias_add (N, ). ";
    CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k)
        << "Weight shape should be equal to K. Assume (M, K) matmul (N, K, transpose_b=True) "
           "bias_add (N, ). ";

    cublas_aux_ld = n;
    // Set Middle result shape.
    long cublas_aligned_aux_ld = AlignReluAuxLd(cublas_aux_ld);
    int64_t aux_size = cublas_aligned_aux_ld / 32;  // Cause we use int32_t as dtype
    ctx->SetOutputShape("cublas_aux", idx, Shape({m, aux_size}));
    ctx->SetOutputShape("hidden", idx, Shape({m, n}));
    // Set for next layer.
    k = n;
  }
  ctx->SetOutputShape("out", 0, Shape({m, n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("x", 0);

  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(in_desc.data_type()) << ", but got "
        << DataType_Name(first_in_desc.data_type());
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(first_in_desc.data_type());

  for (int32_t i = 0; i < ctx->output_size("hidden"); i++) {
    user_op::TensorDesc* hidden_desc = ctx->MutOutputTensorDesc("hidden", i);
    hidden_desc->set_data_type(first_in_desc.data_type());
  }

  for (int32_t i = 0; i < ctx->output_size("cublas_aux"); i++) {
    user_op::TensorDesc* aux_desc = ctx->MutOutputTensorDesc("cublas_aux", i);
    aux_desc->set_data_type(DataType::kInt32);
  }

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedMatmulBiasAddReluDropoutOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmul(ctx);
}

/*static*/ Maybe<void> FusedMatmulBiasAddReluDropoutOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedMatmulBiasAddReluDropoutOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0);
  for (int i = 0; i < ctx->user_op_conf().input_size("weights"); ++i) {
    builder.Broadcast(user_op::OpArg("weights", i));
  }
  for (int i = 0; i < ctx->user_op_conf().input_size("biases"); ++i) {
    builder.Broadcast(user_op::OpArg("biases", i));
  }
  for (int i = 0; i < ctx->user_op_conf().output_size("cublas_aux"); ++i) {
    builder.Split(user_op::OpArg("cublas_aux", i), 0);
  }
  for (int i = 0; i < ctx->user_op_conf().output_size("hidden"); ++i) {
    builder.Split(user_op::OpArg("hidden", i), 0);
  }
  builder.Split(user_op::OpArg("out", 0), 0);
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedMatmulBiasAddReluDropoutOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

}  // namespace oneflow
