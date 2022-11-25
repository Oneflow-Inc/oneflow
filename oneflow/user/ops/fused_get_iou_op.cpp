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

Maybe<void> FusedGetIouOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);

  user_op::TensorDesc* iou = ctx->MutOutputTensorDesc("iou", 0);
  iou->set_is_dynamic(w1.is_dynamic());
  iou->set_shape(w1.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetIouOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetIouOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetIouOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->InputTensorDesc("w1", 0);

  user_op::TensorDesc* iou = ctx->MutOutputTensorDesc("iou", 0);
  iou->set_data_type(w1.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetIouOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& w1 = ctx->LogicalTensorDesc4InputArgNameAndIndex("w1", 0);
  FOR_RANGE(int64_t, i, 0, w1.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("w1", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("w2", 0), i)
        .Split(user_op::OpArg("h2", 0), i)
        .Split(user_op::OpArg("inter", 0), i)
        .Split(user_op::OpArg("iou", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetIouGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& diou = ctx->InputTensorDesc("diou", 0);

  user_op::TensorDesc* dw1 = ctx->MutOutputTensorDesc("dw1", 0);
  dw1->set_is_dynamic(diou.is_dynamic());
  dw1->set_shape(diou.shape());

  user_op::TensorDesc* dh1 = ctx->MutOutputTensorDesc("dh1", 0);
  dh1->set_is_dynamic(diou.is_dynamic());
  dh1->set_shape(diou.shape());

  user_op::TensorDesc* dinter = ctx->MutOutputTensorDesc("dinter", 0);
  dinter->set_is_dynamic(diou.is_dynamic());
  dinter->set_shape(diou.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetIouGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetIouGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetIouGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& diou = ctx->InputTensorDesc("diou", 0);

  user_op::TensorDesc* dw1 = ctx->MutOutputTensorDesc("dw1", 0);
  dw1->set_data_type(diou.data_type());

  user_op::TensorDesc* dh1 = ctx->MutOutputTensorDesc("dh1", 0);
  dh1->set_data_type(diou.data_type());

  user_op::TensorDesc* dinter = ctx->MutOutputTensorDesc("dinter", 0);
  dinter->set_data_type(diou.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetIouGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  FOR_RANGE(int64_t, i, 0, dy.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("diou", 0), i)
        .Split(user_op::OpArg("w1", 0), i)
        .Split(user_op::OpArg("h1", 0), i)
        .Split(user_op::OpArg("w2", 0), i)
        .Split(user_op::OpArg("h2", 0), i)
        .Split(user_op::OpArg("inter", 0), i)
        .Split(user_op::OpArg("dw1", 0), i)
        .Split(user_op::OpArg("dh1", 0), i)
        .Split(user_op::OpArg("dinter", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
