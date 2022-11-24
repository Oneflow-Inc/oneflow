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

Maybe<void> LerpOp::InferLogicalTensorDesc(user_op::InferContext *ctx) {
  const user_op::TensorDesc& start = ctx->InputTensorDesc("start", 0);
  const user_op::TensorDesc& end = ctx->InputTensorDesc("end", 0);
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);

  CHECK_EQ_OR_RETURN(start.shape(), end.shape());
  CHECK_EQ_OR_RETURN(start.shape(), weight.shape());

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_is_dynamic(start.is_dynamic());
  out->set_shape(start.shape());

  return Maybe<void>::Ok();
}

Maybe<void> LerpOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return LerpOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> LerpOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& start = ctx->InputTensorDesc("start", 0);
  const user_op::TensorDesc& end = ctx->InputTensorDesc("end", 0);
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);

  CHECK_EQ_OR_RETURN(start.data_type(), end.data_type());
  CHECK_EQ_OR_RETURN(start.data_type(), weight.data_type());

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(start.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> LerpOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& start = ctx->LogicalTensorDesc4InputArgNameAndIndex("start", 0);
  FOR_RANGE(int64_t, i, 0, start.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("start", 0), i)
        .Split(user_op::OpArg("end", 0), i)
        .Split(user_op::OpArg("weight", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> LerpGradOp::InferLogicalTensorDesc(user_op::InferContext *ctx) {
  const user_op::TensorDesc& start = ctx->InputTensorDesc("start", 0);
  const user_op::TensorDesc& end = ctx->InputTensorDesc("end", 0);
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& out_diff = ctx->InputTensorDesc("out_diff", 0);

  CHECK_EQ_OR_RETURN(start.shape(), end.shape());
  CHECK_EQ_OR_RETURN(start.shape(), weight.shape());
  CHECK_EQ_OR_RETURN(start.shape(), out_diff.shape());

  user_op::TensorDesc* start_diff = ctx->MutOutputTensorDesc("start_diff", 0);
  user_op::TensorDesc* end_diff = ctx->MutOutputTensorDesc("end_diff", 0);
  user_op::TensorDesc* weight_diff = ctx->MutOutputTensorDesc("weight_diff", 0);
  start_diff->set_is_dynamic(start.is_dynamic());
  start_diff->set_shape(start.shape());

  end_diff->set_is_dynamic(start.is_dynamic());
  end_diff->set_shape(start.shape());

  weight_diff->set_is_dynamic(start.is_dynamic());
  weight_diff->set_shape(start.shape());

  return Maybe<void>::Ok();
}

Maybe<void> LerpGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return LerpGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> LerpGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& start = ctx->InputTensorDesc("start", 0);
  const user_op::TensorDesc& end = ctx->InputTensorDesc("end", 0);
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& out_diff = ctx->InputTensorDesc("out_diff", 0);

  CHECK_EQ_OR_RETURN(start.data_type(), end.data_type());
  CHECK_EQ_OR_RETURN(start.data_type(), weight.data_type());
  CHECK_EQ_OR_RETURN(start.data_type(), out_diff.data_type());

  user_op::TensorDesc* start_diff = ctx->MutOutputTensorDesc("start_diff", 0);
  user_op::TensorDesc* end_diff = ctx->MutOutputTensorDesc("end_diff", 0);
  user_op::TensorDesc* weight_diff = ctx->MutOutputTensorDesc("weight_diff", 0);
  start_diff->set_data_type(start.data_type());

  end_diff->set_data_type(start.data_type());

  weight_diff->set_data_type(start.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> LerpGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& start = ctx->LogicalTensorDesc4InputArgNameAndIndex("start", 0);
  FOR_RANGE(int64_t, i, 0, start.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("start", 0), i)
        .Split(user_op::OpArg("end", 0), i)
        .Split(user_op::OpArg("weight", 0), i)
        .Split(user_op::OpArg("out_diff", 0), i)
        .Split(user_op::OpArg("start_diff", 0), i)
        .Split(user_op::OpArg("end_diff", 0), i)
        .Split(user_op::OpArg("weight_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
