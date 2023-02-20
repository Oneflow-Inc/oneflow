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

Maybe<void> FusedGetBounddingBoxesCoordOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1 = ctx->InputTensorDesc("x1", 0);
  Shape x1_shape = x1.shape();

  user_op::TensorDesc* b1_x1 = ctx->MutOutputTensorDesc("b1_x1", 0);
  b1_x1->set_is_dynamic(x1.is_dynamic());
  b1_x1->set_shape(x1_shape);

  user_op::TensorDesc* b1_x2 = ctx->MutOutputTensorDesc("b1_x2", 0);
  b1_x2->set_is_dynamic(x1.is_dynamic());
  b1_x2->set_shape(x1_shape);

  user_op::TensorDesc* b1_y1 = ctx->MutOutputTensorDesc("b1_y1", 0);
  b1_y1->set_is_dynamic(x1.is_dynamic());
  b1_y1->set_shape(x1_shape);

  user_op::TensorDesc* b1_y2 = ctx->MutOutputTensorDesc("b1_y2", 0);
  b1_y2->set_is_dynamic(x1.is_dynamic());
  b1_y2->set_shape(x1_shape);

  user_op::TensorDesc* b2_x1 = ctx->MutOutputTensorDesc("b2_x1", 0);
  b2_x1->set_is_dynamic(x1.is_dynamic());
  b2_x1->set_shape(x1_shape);

  user_op::TensorDesc* b2_x2 = ctx->MutOutputTensorDesc("b2_x2", 0);
  b2_x2->set_is_dynamic(x1.is_dynamic());
  b2_x2->set_shape(x1_shape);

  user_op::TensorDesc* b2_y1 = ctx->MutOutputTensorDesc("b2_y1", 0);
  b2_y1->set_is_dynamic(x1.is_dynamic());
  b2_y1->set_shape(x1_shape);

  user_op::TensorDesc* b2_y2 = ctx->MutOutputTensorDesc("b2_y2", 0);
  b2_y2->set_is_dynamic(x1.is_dynamic());
  b2_y2->set_shape(x1_shape);

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetBounddingBoxesCoordOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetBounddingBoxesCoordOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetBounddingBoxesCoordOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x1 = ctx->InputTensorDesc("x1", 0);

  user_op::TensorDesc* b1_x1 = ctx->MutOutputTensorDesc("b1_x1", 0);
  b1_x1->set_data_type(x1.data_type());

  user_op::TensorDesc* b1_x2 = ctx->MutOutputTensorDesc("b1_x2", 0);
  b1_x2->set_data_type(x1.data_type());

  user_op::TensorDesc* b1_y1 = ctx->MutOutputTensorDesc("b1_y1", 0);
  b1_y1->set_data_type(x1.data_type());

  user_op::TensorDesc* b1_y2 = ctx->MutOutputTensorDesc("b1_y2", 0);
  b1_y2->set_data_type(x1.data_type());

  user_op::TensorDesc* b2_x1 = ctx->MutOutputTensorDesc("b2_x1", 0);
  b2_x1->set_data_type(x1.data_type());

  user_op::TensorDesc* b2_x2 = ctx->MutOutputTensorDesc("b2_x2", 0);
  b2_x2->set_data_type(x1.data_type());

  user_op::TensorDesc* b2_y1 = ctx->MutOutputTensorDesc("b2_y1", 0);
  b2_y1->set_data_type(x1.data_type());

  user_op::TensorDesc* b2_y2 = ctx->MutOutputTensorDesc("b2_y2", 0);
  b2_y2->set_data_type(x1.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetBounddingBoxesCoordOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("x1", 0);
  FOR_RANGE(int64_t, i, 0, x1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x1", 0), i)
        .Split(user_op::OpArg("y1", 0), i)
        .Split(user_op::OpArg("w1", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("x2", 0), i)
        .Split(user_op::OpArg("y2", 0), i)
        .Split(user_op::OpArg("w2", 0), i)
        .Split(user_op::OpArg("h2", 0), i)
        .Split(user_op::OpArg("b1_x1", 0), i)
        .Split(user_op::OpArg("b1_x2", 0), i)
        .Split(user_op::OpArg("b1_y1", 0), i)
        .Split(user_op::OpArg("b1_y2", 0), i)
        .Split(user_op::OpArg("b2_x1", 0), i)
        .Split(user_op::OpArg("b2_x2", 0), i)
        .Split(user_op::OpArg("b2_y1", 0), i)
        .Split(user_op::OpArg("b2_y2", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetBounddingBoxesCoordGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1_diff = ctx->InputTensorDesc("b1_x1_diff", 0);

  user_op::TensorDesc* x1_diff = ctx->MutOutputTensorDesc("x1_diff", 0);
  x1_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  x1_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* y1_diff = ctx->MutOutputTensorDesc("y1_diff", 0);
  y1_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  y1_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* w1_diff = ctx->MutOutputTensorDesc("w1_diff", 0);
  w1_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  w1_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* h1_diff = ctx->MutOutputTensorDesc("h1_diff", 0);
  h1_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  h1_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* x2_diff = ctx->MutOutputTensorDesc("x2_diff", 0);
  x2_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  x2_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* y2_diff = ctx->MutOutputTensorDesc("y2_diff", 0);
  y2_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  y2_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* w2_diff = ctx->MutOutputTensorDesc("w2_diff", 0);
  w2_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  w2_diff->set_shape(b1_x1_diff.shape());

  user_op::TensorDesc* h2_diff = ctx->MutOutputTensorDesc("h2_diff", 0);
  h2_diff->set_is_dynamic(b1_x1_diff.is_dynamic());
  h2_diff->set_shape(b1_x1_diff.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetBounddingBoxesCoordGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetBounddingBoxesCoordGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetBounddingBoxesCoordGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& b1_x1_diff = ctx->InputTensorDesc("b1_x1_diff", 0);

  user_op::TensorDesc* x1_diff = ctx->MutOutputTensorDesc("x1_diff", 0);
  x1_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* y1_diff = ctx->MutOutputTensorDesc("y1_diff", 0);
  y1_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* w1_diff = ctx->MutOutputTensorDesc("w1_diff", 0);
  w1_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* h1_diff = ctx->MutOutputTensorDesc("h1_diff", 0);
  h1_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* x2_diff = ctx->MutOutputTensorDesc("x2_diff", 0);
  x2_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* y2_diff = ctx->MutOutputTensorDesc("y2_diff", 0);
  y2_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* w2_diff = ctx->MutOutputTensorDesc("w2_diff", 0);
  w2_diff->set_data_type(b1_x1_diff.data_type());

  user_op::TensorDesc* h2_diff = ctx->MutOutputTensorDesc("h2_diff", 0);
  h2_diff->set_data_type(b1_x1_diff.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetBounddingBoxesCoordGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& b1_x1_diff =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("b1_x1_diff", 0);
  FOR_RANGE(int64_t, i, 0, b1_x1_diff.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("b1_x1_diff", 0), i)
        .Split(user_op::OpArg("b1_x2_diff", 0), i)
        .Split(user_op::OpArg("b1_y1_diff", 0), i)
        .Split(user_op::OpArg("b1_y2_diff", 0), i)
        .Split(user_op::OpArg("b2_x1_diff", 0), i)
        .Split(user_op::OpArg("b2_x2_diff", 0), i)
        .Split(user_op::OpArg("b2_y1_diff", 0), i)
        .Split(user_op::OpArg("b2_y2_diff", 0), i)
        .Split(user_op::OpArg("x1_diff", 0), i)
        .Split(user_op::OpArg("y1_diff", 0), i)
        .Split(user_op::OpArg("w1_diff", 0), i)
        .Split(user_op::OpArg("h1_diff", 0), i)
        .Split(user_op::OpArg("x2_diff", 0), i)
        .Split(user_op::OpArg("y2_diff", 0), i)
        .Split(user_op::OpArg("w2_diff", 0), i)
        .Split(user_op::OpArg("h2_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
