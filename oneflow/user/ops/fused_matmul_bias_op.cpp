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
  x: (m_i, ... m_1, k)
  weight: (n, k) need transpose
  bias: (n)
  */

  CHECK_GE_OR_RETURN(x_desc.shape().NumAxes(), 2);
  const int64_t k = x_desc.shape().At(x_desc.shape().NumAxes() - 1);

  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& bias_desc = ctx->InputTensorDesc("bias", 0);
  CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(bias_desc.shape().NumAxes(), 1);

  const int64_t n = weight_desc.shape().At(0);

  CHECK_EQ_OR_RETURN(bias_desc.shape().At(0), n);
  CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k);

  Shape out_shape = x_desc.shape();
  out_shape[x_desc.shape().NumAxes() - 1] = n;
  ctx->SetOutputShape("out", 0, out_shape);

  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& _add_to_output_desc = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(_add_to_output_desc.shape(), out_shape);
  }

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

  if (ctx->has_input("_add_to_output", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("_add_to_output", 0), out_desc->data_type())
        << "InferDataType Failed. _add_to_output Expected " << DataType_Name(out_desc->data_type())
        << ", but got " << DataType_Name(ctx->InputDType("_add_to_output", 0));
  }

  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedMatmulBiasOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBias(ctx);
}

/*static*/ Maybe<void> FusedMatmulBiasOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedMatmulBiasOp::GetSbp(user_op::SbpContext* ctx) {
  // Currently Only support S0 B B B B ... S0
  auto builder = ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0);
  builder.Broadcast(user_op::OpArg("weight", 0));
  builder.Broadcast(user_op::OpArg("bias", 0));
  builder.Split(user_op::OpArg("out", 0), 0);
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedMatmulBiasOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBias(ctx);
}

}  // namespace oneflow
