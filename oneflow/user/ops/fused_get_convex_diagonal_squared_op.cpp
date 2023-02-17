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

Maybe<void> FusedGetConvexDiagonalSquaredOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->InputTensorDesc("b1_x1", 0);

  user_op::TensorDesc* c2 = ctx->MutOutputTensorDesc("c2", 0);
  c2->set_is_dynamic(b1_x1.is_dynamic());
  c2->set_shape(b1_x1.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetConvexDiagonalSquaredOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetConvexDiagonalSquaredOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetConvexDiagonalSquaredOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->InputTensorDesc("b1_x1", 0);

  user_op::TensorDesc* c2 = ctx->MutOutputTensorDesc("c2", 0);
  c2->set_data_type(b1_x1.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetConvexDiagonalSquaredOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("b1_x1", 0);
  FOR_RANGE(int64_t, i, 0, b1_x1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("b1_x1", 0), i)
        .Split(user_op::OpArg("b1_x2", 0), i)
        .Split(user_op::OpArg("b2_x1", 0), i)
        .Split(user_op::OpArg("b2_x2", 0), i)
        .Split(user_op::OpArg("b1_y1", 0), i)
        .Split(user_op::OpArg("b1_y2", 0), i)
        .Split(user_op::OpArg("b2_y1", 0), i)
        .Split(user_op::OpArg("b2_y2", 0), i)
        .Split(user_op::OpArg("c2", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetConvexDiagonalSquaredGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->InputTensorDesc("b1_x1", 0);

  user_op::TensorDesc* b1_x1_diff = ctx->MutOutputTensorDesc("b1_x1_diff", 0);
  b1_x1_diff->set_is_dynamic(b1_x1.is_dynamic());
  b1_x1_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b1_x2_diff = ctx->MutOutputTensorDesc("b1_x2_diff", 0);
  b1_x2_diff->set_is_dynamic(b1_x1.is_dynamic());
  b1_x2_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b2_x1_diff = ctx->MutOutputTensorDesc("b2_x1_diff", 0);
  b2_x1_diff->set_is_dynamic(b1_x1.is_dynamic());
  b2_x1_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b2_x2_diff = ctx->MutOutputTensorDesc("b2_x2_diff", 0);
  b2_x2_diff->set_is_dynamic(b1_x1.is_dynamic());
  b2_x2_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b1_y1_diff = ctx->MutOutputTensorDesc("b1_y1_diff", 0);
  b1_y1_diff->set_is_dynamic(b1_x1.is_dynamic());
  b1_y1_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b1_y2_diff = ctx->MutOutputTensorDesc("b1_y2_diff", 0);
  b1_y2_diff->set_is_dynamic(b1_x1.is_dynamic());
  b1_y2_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b2_y1_diff = ctx->MutOutputTensorDesc("b2_y1_diff", 0);
  b2_y1_diff->set_is_dynamic(b1_x1.is_dynamic());
  b2_y1_diff->set_shape(b1_x1.shape());

  user_op::TensorDesc* b2_y2_diff = ctx->MutOutputTensorDesc("b2_y2_diff", 0);
  b2_y2_diff->set_is_dynamic(b1_x1.is_dynamic());
  b2_y2_diff->set_shape(b1_x1.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetConvexDiagonalSquaredGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return FusedGetConvexDiagonalSquaredGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetConvexDiagonalSquaredGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->InputTensorDesc("b1_x1", 0);

  user_op::TensorDesc* b1_x1_diff = ctx->MutOutputTensorDesc("b1_x1_diff", 0);
  b1_x1_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b1_x2_diff = ctx->MutOutputTensorDesc("b1_x2_diff", 0);
  b1_x2_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b2_x1_diff = ctx->MutOutputTensorDesc("b2_x1_diff", 0);
  b2_x1_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b2_x2_diff = ctx->MutOutputTensorDesc("b2_x2_diff", 0);
  b2_x2_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b1_y1_diff = ctx->MutOutputTensorDesc("b1_y1_diff", 0);
  b1_y1_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b1_y2_diff = ctx->MutOutputTensorDesc("b1_y2_diff", 0);
  b1_y2_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b2_y1_diff = ctx->MutOutputTensorDesc("b2_y1_diff", 0);
  b2_y1_diff->set_data_type(b1_x1.data_type());

  user_op::TensorDesc* b2_y2_diff = ctx->MutOutputTensorDesc("b2_y2_diff", 0);
  b2_y2_diff->set_data_type(b1_x1.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetConvexDiagonalSquaredGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& b1_x1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("b1_x1", 0);
  FOR_RANGE(int64_t, i, 0, b1_x1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("c2_diff", 0), i)
        .Split(user_op::OpArg("b1_x1", 0), i)
        .Split(user_op::OpArg("b1_x2", 0), i)
        .Split(user_op::OpArg("b2_x1", 0), i)
        .Split(user_op::OpArg("b2_x2", 0), i)
        .Split(user_op::OpArg("b1_y1", 0), i)
        .Split(user_op::OpArg("b1_y2", 0), i)
        .Split(user_op::OpArg("b2_y1", 0), i)
        .Split(user_op::OpArg("b2_y2", 0), i)
        .Split(user_op::OpArg("b1_x1_diff", 0), i)
        .Split(user_op::OpArg("b1_x2_diff", 0), i)
        .Split(user_op::OpArg("b2_x1_diff", 0), i)
        .Split(user_op::OpArg("b2_x2_diff", 0), i)
        .Split(user_op::OpArg("b1_y1_diff", 0), i)
        .Split(user_op::OpArg("b1_y2_diff", 0), i)
        .Split(user_op::OpArg("b2_y1_diff", 0), i)
        .Split(user_op::OpArg("b2_y2_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
