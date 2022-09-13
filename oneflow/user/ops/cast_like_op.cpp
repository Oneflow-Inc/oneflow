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

/* static */ Maybe<void> CastLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CastLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CastLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  for (int i = 0; i < in_shape.NumAxes(); ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Split(user_op::OpArg("dtype_like", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dtype_like", 0))
      .Broadcast(user_op::OpArg("in", 0))
      .Broadcast(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dtype_like", 0))
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dtype_like", 0))
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CastLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* dtype_like_modifier = GetInputArgModifierFn("dtype_like", 0);
  CHECK_NOTNULL_OR_RETURN(dtype_like_modifier);
  dtype_like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CastLikeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dtype_like_tensor_desc = ctx->InputTensorDesc("dtype_like", 0);
  user_op::TensorDesc* output_tensor_desc = ctx->MutOutputTensorDesc("out", 0);
  output_tensor_desc->set_data_type(dtype_like_tensor_desc.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
