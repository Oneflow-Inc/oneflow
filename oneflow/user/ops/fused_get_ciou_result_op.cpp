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

Maybe<void> FusedGetCiouResultOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);

  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_is_dynamic(v.is_dynamic());
  y->set_shape(v.shape());

  user_op::TensorDesc* ahpha = ctx->MutOutputTensorDesc("alpha", 0);
  ahpha->set_is_dynamic(v.is_dynamic());
  ahpha->set_shape(v.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouResultOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetCiouResultOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetCiouResultOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& v = ctx->InputTensorDesc("v", 0);

  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_data_type(v.data_type());

  user_op::TensorDesc* alpha = ctx->MutOutputTensorDesc("alpha", 0);
  alpha->set_data_type(v.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouResultOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& v = ctx->LogicalTensorDesc4InputArgNameAndIndex("v", 0);
  FOR_RANGE(int64_t, i, 0, v.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("v", 0), i)
        .Split(user_op::OpArg("iou", 0), i)
        .Split(user_op::OpArg("rho2", 0), i)
        .Split(user_op::OpArg("c2", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("alpha", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouResultGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);

  user_op::TensorDesc* dv = ctx->MutOutputTensorDesc("dv", 0);
  dv->set_is_dynamic(dy.is_dynamic());
  dv->set_shape(dy.shape());

  user_op::TensorDesc* diou = ctx->MutOutputTensorDesc("diou", 0);
  diou->set_is_dynamic(dy.is_dynamic());
  diou->set_shape(dy.shape());

  user_op::TensorDesc* drho2 = ctx->MutOutputTensorDesc("drho2", 0);
  drho2->set_is_dynamic(dy.is_dynamic());
  drho2->set_shape(dy.shape());

  user_op::TensorDesc* dc2 = ctx->MutOutputTensorDesc("dc2", 0);
  dc2->set_is_dynamic(dy.is_dynamic());
  dc2->set_shape(dy.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouResultGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetCiouResultGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetCiouResultGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);

  user_op::TensorDesc* dv = ctx->MutOutputTensorDesc("dv", 0);
  dv->set_data_type(dy.data_type());

  user_op::TensorDesc* diou = ctx->MutOutputTensorDesc("diou", 0);
  diou->set_data_type(dy.data_type());

  user_op::TensorDesc* drho2 = ctx->MutOutputTensorDesc("drho2", 0);
  drho2->set_data_type(dy.data_type());

  user_op::TensorDesc* dc2 = ctx->MutOutputTensorDesc("dc2", 0);
  dc2->set_data_type(dy.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetCiouResultGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  FOR_RANGE(int64_t, i, 0, dy.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("alpha", 0), i)
        .Split(user_op::OpArg("rho2", 0), i)
        .Split(user_op::OpArg("c2", 0), i)
        .Split(user_op::OpArg("dv", 0), i)
        .Split(user_op::OpArg("diou", 0), i)
        .Split(user_op::OpArg("drho2", 0), i)
        .Split(user_op::OpArg("dc2", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
