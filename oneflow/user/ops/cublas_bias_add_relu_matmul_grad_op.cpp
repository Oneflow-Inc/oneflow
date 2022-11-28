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
  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  const int64_t bias_size = weight_desc.shape().At(1);
  Shape d_grad_shape({dy_desc.shape().At(0), weight_desc.shape().At(1)});
  ctx->SetOutputShape("d_grad", 0, d_grad_shape);
  ctx->SetOutputShape("d_bias", 0, Shape({bias_size}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBackward(user_op::InferContext* ctx) {
  const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(weight_desc.data_type(), dy_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(dy_desc.data_type()) << ", but got "
      << DataType_Name(weight_desc.data_type());

  user_op::TensorDesc* d_grad_desc = ctx->MutOutputTensorDesc("d_grad", 0);
  user_op::TensorDesc* d_bias_desc = ctx->MutOutputTensorDesc("d_bias", 0);

  d_grad_desc->set_data_type(dy_desc.data_type());
  d_bias_desc->set_data_type(dy_desc.data_type());
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> CublasBiasAddReluMatmulGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBackward(ctx);
}

/*static*/ Maybe<void> CublasBiasAddReluMatmulGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CublasBiasAddReluMatmulGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("aux", 0), 0)
      .Split(user_op::OpArg("d_grad", 0), 0)
      .PartialSum(user_op::OpArg("d_bias", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasBiasAddReluMatmulGradOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBackward(ctx);
}

}  // namespace oneflow
