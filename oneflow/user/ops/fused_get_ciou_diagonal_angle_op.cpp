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

Maybe<void> FusedGetCiouDiagonalAngleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);
  const user_op::TensorDesc& h1 = ctx->InputTensorDesc("h1", 0);
  const user_op::TensorDesc& w2 = ctx->InputTensorDesc("w2", 0);
  const user_op::TensorDesc& h2 = ctx->InputTensorDesc("h2", 0);

  CHECK_EQ_OR_RETURN(w1.shape(), h1.shape());
  CHECK_EQ_OR_RETURN(w1.shape(), w2.shape());
  CHECK_EQ_OR_RETURN(w1.shape(), h2.shape());

  user_op::TensorDesc* v = ctx->MutOutputTensorDesc("v", 0);
  v->set_is_dynamic(w1.is_dynamic());
  v->set_shape(w1.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouDiagonalAngleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetCiouDiagonalAngleOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetCiouDiagonalAngleOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);
  const user_op::TensorDesc& h1 = ctx->InputTensorDesc("h1", 0);
  const user_op::TensorDesc& w2 = ctx->InputTensorDesc("w2", 0);
  const user_op::TensorDesc& h2 = ctx->InputTensorDesc("h2", 0);

  CHECK_EQ_OR_RETURN(w1.data_type(), h1.data_type());
  CHECK_EQ_OR_RETURN(w1.data_type(), w2.data_type());
  CHECK_EQ_OR_RETURN(w1.data_type(), h2.data_type());

  user_op::TensorDesc* v = ctx->MutOutputTensorDesc("v", 0);
  v->set_data_type(w1.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouDiagonalAngleOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("w1", 0);
  FOR_RANGE(int64_t, i, 0, b1_x1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("w1", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("w2", 0), i)
        .Split(user_op::OpArg("h2", 0), i)
        .Split(user_op::OpArg("v", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouDiagonalAngleGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);
  const user_op::TensorDesc& h1 = ctx->InputTensorDesc("h1", 0);
  const user_op::TensorDesc& w2 = ctx->InputTensorDesc("w2", 0);
  const user_op::TensorDesc& h2 = ctx->InputTensorDesc("h2", 0);
  const user_op::TensorDesc& v_diff = ctx->InputTensorDesc("v_diff", 0);

  CHECK_EQ_OR_RETURN(w1.shape(), h1.shape());
  CHECK_EQ_OR_RETURN(w1.shape(), w2.shape());
  CHECK_EQ_OR_RETURN(w1.shape(), h2.shape());
  CHECK_EQ_OR_RETURN(w1.shape(), v_diff.shape());

  user_op::TensorDesc* w1_diff = ctx->MutOutputTensorDesc("w1_diff", 0);
  w1_diff->set_is_dynamic(w1.is_dynamic());
  w1_diff->set_shape(w1.shape());

  user_op::TensorDesc* h1_diff = ctx->MutOutputTensorDesc("h1_diff", 0);
  h1_diff->set_is_dynamic(w1.is_dynamic());
  h1_diff->set_shape(w1.shape());

  user_op::TensorDesc* w2_diff = ctx->MutOutputTensorDesc("w2_diff", 0);
  w2_diff->set_is_dynamic(w1.is_dynamic());
  w2_diff->set_shape(w1.shape());

  user_op::TensorDesc* h2_diff = ctx->MutOutputTensorDesc("h2_diff", 0);
  h2_diff->set_is_dynamic(w1.is_dynamic());
  h2_diff->set_shape(w1.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouDiagonalAngleGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetCiouDiagonalAngleGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetCiouDiagonalAngleGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);
  const user_op::TensorDesc& h1 = ctx->InputTensorDesc("h1", 0);
  const user_op::TensorDesc& w2 = ctx->InputTensorDesc("w2", 0);
  const user_op::TensorDesc& h2 = ctx->InputTensorDesc("h2", 0);
  const user_op::TensorDesc& v_diff = ctx->InputTensorDesc("v_diff", 0);

  CHECK_EQ_OR_RETURN(w1.data_type(), h1.data_type());
  CHECK_EQ_OR_RETURN(w1.data_type(), w2.data_type());
  CHECK_EQ_OR_RETURN(w1.data_type(), h2.data_type());
  CHECK_EQ_OR_RETURN(w1.data_type(), v_diff.data_type());

  user_op::TensorDesc* w1_diff = ctx->MutOutputTensorDesc("w1_diff", 0);
  w1_diff->set_is_dynamic(w1.is_dynamic());
  w1_diff->set_data_type(w1.data_type());

  user_op::TensorDesc* h1_diff = ctx->MutOutputTensorDesc("h1_diff", 0);
  h1_diff->set_is_dynamic(w1.is_dynamic());
  h1_diff->set_data_type(w1.data_type());

  user_op::TensorDesc* w2_diff = ctx->MutOutputTensorDesc("w2_diff", 0);
  w2_diff->set_is_dynamic(w1.is_dynamic());
  w2_diff->set_data_type(w1.data_type());

  user_op::TensorDesc* h2_diff = ctx->MutOutputTensorDesc("h2_diff", 0);
  h2_diff->set_is_dynamic(w1.is_dynamic());
  h2_diff->set_data_type(w1.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouDiagonalAngleGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("w1", 0);
  FOR_RANGE(int64_t, i, 0, w1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("w1", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("w2", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("v_diff", 0), i)
        .Split(user_op::OpArg("w1_diff", 0), i)
        .Split(user_op::OpArg("h1_diff", 0), i)
        .Split(user_op::OpArg("w2_diff", 0), i)
        .Split(user_op::OpArg("h2_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
