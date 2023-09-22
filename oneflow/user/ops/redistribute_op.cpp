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

/* static */ Maybe<void> RedistributeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in.shape().NumAxes(), 2);
  const int n = in.shape().At(0);
  const int k = in.shape().At(1);

  CHECK_EQ_OR_RETURN(k % 16, 0);

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  out->set_shape(ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RedistributeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RedistributeOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Split(user_op::OpArg("out", 0), 0).Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RedistributeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputTensorDesc("in", 0).data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
