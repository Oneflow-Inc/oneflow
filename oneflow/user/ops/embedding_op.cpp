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

/* static */ Maybe<void> EmbeddingRenormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingRenormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> EmbeddingRenormOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingRenormOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const Shape& indices_shape = ctx->InputShape("indices", 0);

  DimVector out_dim_vec;
  out_dim_vec.insert(out_dim_vec.end(), indices_shape.dim_vec().cbegin(),
                     indices_shape.dim_vec().cend());
  out_dim_vec.push_back(weight_shape.At(1));

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> EmbeddingOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t indices_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
  const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");

  if (!scale_grad_by_freq) {
    FOR_RANGE(int64_t, i, 0, indices_num_axes) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("indices", 0), i)
          .Broadcast(user_op::OpArg("weight", 0))
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("weight", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_shape(weight_shape);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return EmbeddingGradOp::InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> EmbeddingGradOp::GetSbp(user_op::SbpContext* ctx) {
  const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");
  const int64_t indices_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();

  if (!scale_grad_by_freq) {
    for (int32_t i = 0; i < indices_num_axes; i++) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), i)
          .Broadcast(user_op::OpArg("weight", 0))
          .Split(user_op::OpArg("indices", 0), i)
          .PartialSum(user_op::OpArg("dx", 0))
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("weight", 0), ctx->InputDType("dy", 0))
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("dy", 0))
      << ", but got " << DataType_Name(ctx->InputDType("weight", 0));
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
