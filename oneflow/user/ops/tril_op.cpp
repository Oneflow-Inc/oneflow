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

/*static*/ Maybe<void> TrilOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  bool fill_zero = ctx->Attr<bool>("is_floating_fill_value")
                       ? (ctx->Attr<double>("floating_fill_value") == static_cast<double>(0))
                       : (ctx->Attr<int64_t>("integer_fill_value") == static_cast<int64_t>(0));
  if (fill_zero) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("in", 0))
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TrilOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_GE_OR_RETURN(in.shape().NumAxes(), 2);
  out->set_shape(in.shape());
  out->set_is_dynamic(in.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TrilOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TrilOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedScaleTrilOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  bool fill_zero = ctx->Attr<bool>("is_floating_fill_value")
                       ? (ctx->Attr<double>("floating_fill_value") == static_cast<double>(0))
                       : (ctx->Attr<int64_t>("integer_fill_value") == static_cast<int64_t>(0));
  if (fill_zero) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("in", 0))
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> FusedScaleTrilOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_GE_OR_RETURN(in.shape().NumAxes(), 2);
  out->set_shape(in.shape());
  out->set_is_dynamic(in.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> FusedScaleTrilOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> FusedScaleTrilOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
