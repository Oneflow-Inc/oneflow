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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedMatmulBias(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  /*
  x: (m, k)
  weight: (n, k) need transpose
  bias: (n)
  */
  int64_t m = 0, n = 0, k = 0;
  m = x_desc.shape().At(0);
  k = x_desc.shape().At(1);

  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& bias_desc = ctx->InputTensorDesc("bias", 0);
  CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(bias_desc.shape().NumAxes(), 1);

  n = weight_desc.shape().At(0);
  CHECK_EQ_OR_RETURN(bias_desc.shape().At(0), n);
  CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k);

  ctx->SetOutputShape("out", 0, Shape({m, n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBias(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("x", 0);

  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(first_in_desc.data_type())
        << ", but got " << DataType_Name(in_desc.data_type());
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(first_in_desc.data_type());

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CublasFusedMatmulAddBiasOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBias(ctx);
}

/*static*/ Maybe<void> CublasFusedMatmulAddBiasOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CublasFusedMatmulAddBiasOp::GetSbp(user_op::SbpContext* ctx) {
  // Currently Only support S0 B B B B ... S0
  auto builder = ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0);
  builder.Broadcast(user_op::OpArg("weight", 0));
  builder.Broadcast(user_op::OpArg("bias", 0));
  builder.Split(user_op::OpArg("out", 0), 0);
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasFusedMatmulAddBiasOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBias(ctx);
}

}  // namespace oneflow
