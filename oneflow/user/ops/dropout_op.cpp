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

/* static */ Maybe<void> DropoutOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  ctx->SetOutputShape("out", 0, in_shape);
  ctx->SetOutputShape("mask", 0, in_shape);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DropoutOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DropoutOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                              const user_op::UserOpConfWrapper& conf) {
  float rate = conf.attr<float>("rate");
  CHECK_GE_OR_RETURN(rate, 0.0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("mask", 0, DataType::kBool);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  ctx->SetOutputShape("dx", 0, dy_shape);
  ctx->SetOutputIsDynamic("dx", 0, ctx->InputIsDynamic("dy", 0));
  CHECK_EQ_OR_RETURN(ctx->InputShape("mask", 0), dy_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DropoutGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DropoutGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  FOR_RANGE(int64_t, axis, 0, dy_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), axis)
        .Split(user_op::OpArg("mask", 0), axis)
        .Split(user_op::OpArg("dx", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                  const user_op::UserOpConfWrapper& conf) {
  float scale = conf.attr<float>("scale");
  CHECK_GT_OR_RETURN(scale, 1);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  CHECK_EQ_OR_RETURN(ctx->InputDType("mask", 0), DataType::kBool)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kBool) << ", but got "
      << DataType_Name(ctx->InputDType("mask", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RandomMaskLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("like", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RandomMaskLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RandomMaskLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& like_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0);
  FOR_RANGE(int64_t, axis, 0, like_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("like", 0), axis)
        .Split(user_op::OpArg("out", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RandomMaskLikeOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                     const user_op::UserOpConfWrapper& conf) {
  float rate = conf.attr<float>("rate");
  CHECK_GE_OR_RETURN(rate, 0);
  CHECK_LT_OR_RETURN(rate, 1);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RandomMaskLikeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kBool);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
