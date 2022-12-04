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

Maybe<void> FusedGetPboxOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->InputTensorDesc("pxy", 0);
  const user_op::TensorDesc& pwh = ctx->InputTensorDesc("pwh", 0);
  const user_op::TensorDesc& anchors = ctx->InputTensorDesc("anchors", 0);

  const Shape& pxy_shape = pxy.shape();

  CHECK_EQ_OR_RETURN(pxy.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(pxy.shape(), pwh.shape());
  CHECK_EQ_OR_RETURN(pxy.shape(), anchors.shape());

  user_op::TensorDesc* pbox = ctx->MutOutputTensorDesc("pbox", 0);
  pbox->set_is_dynamic(pxy.is_dynamic());
  pbox->set_shape(Shape({pxy_shape.At(0), pxy_shape.At(1) * 2}));

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetPboxOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetPboxOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetPboxOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->InputTensorDesc("pxy", 0);
  const user_op::TensorDesc& pwh = ctx->InputTensorDesc("pwh", 0);
  const user_op::TensorDesc& anchors = ctx->InputTensorDesc("anchors", 0);

  CHECK_EQ_OR_RETURN(pxy.data_type(), pwh.data_type());
  CHECK_EQ_OR_RETURN(pxy.data_type(), anchors.data_type());

  user_op::TensorDesc* pbox = ctx->MutOutputTensorDesc("pbox", 0);
  pbox->set_data_type(pxy.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetPboxOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->LogicalTensorDesc4InputArgNameAndIndex("pxy", 0);
  FOR_RANGE(int64_t, i, 0, pxy.shape().NumAxes()) {
    if (i != 1) {
      ctx->NewBuilder()
        .Split(user_op::OpArg("pxy", 0), i)
        .Split(user_op::OpArg("pwh", 0), i)
        .Split(user_op::OpArg("anchors", 0), i)
        .Split(user_op::OpArg("pbox", 0), i)
        .Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedGetPboxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->InputTensorDesc("pxy", 0);
  const user_op::TensorDesc& pwh = ctx->InputTensorDesc("pwh", 0);
  const user_op::TensorDesc& anchors = ctx->InputTensorDesc("anchors", 0);
  const user_op::TensorDesc& pbox_diff = ctx->InputTensorDesc("pbox_diff", 0);

  CHECK_EQ_OR_RETURN(pxy.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(pxy.shape(), pwh.shape());
  CHECK_EQ_OR_RETURN(pxy.shape(), pwh.shape());
  CHECK_EQ_OR_RETURN(pxy.shape().At(0), pbox_diff.shape().At(0));
  CHECK_EQ_OR_RETURN(pxy.shape().At(1) * 2, pbox_diff.shape().At(1));

  user_op::TensorDesc* pxy_diff = ctx->MutOutputTensorDesc("pxy_diff", 0);
  pxy_diff->set_is_dynamic(pxy.is_dynamic());
  pxy_diff->set_shape(pxy.shape());

  user_op::TensorDesc* pwh_diff = ctx->MutOutputTensorDesc("pwh_diff", 0);
  pwh_diff->set_is_dynamic(pwh.is_dynamic());
  pwh_diff->set_shape(pwh.shape());

  user_op::TensorDesc* anchors_diff = ctx->MutOutputTensorDesc("anchors_diff", 0);
  anchors_diff->set_is_dynamic(anchors.is_dynamic());
  anchors_diff->set_shape(anchors.shape());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetPboxGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedGetPboxGradOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedGetPboxGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->InputTensorDesc("pxy", 0);
  const user_op::TensorDesc& pwh = ctx->InputTensorDesc("pwh", 0);
  const user_op::TensorDesc& anchors = ctx->InputTensorDesc("anchors", 0);
  const user_op::TensorDesc& pbox_diff = ctx->InputTensorDesc("pbox_diff", 0);

  CHECK_EQ_OR_RETURN(pxy.data_type(), pwh.data_type());
  CHECK_EQ_OR_RETURN(pxy.data_type(), anchors.data_type());
  CHECK_EQ_OR_RETURN(pxy.data_type(), pbox_diff.data_type());

  user_op::TensorDesc* pxy_diff = ctx->MutOutputTensorDesc("pxy_diff", 0);
  pxy_diff->set_is_dynamic(pxy.is_dynamic());
  pxy_diff->set_data_type(pxy.data_type());

  user_op::TensorDesc* pwh_diff = ctx->MutOutputTensorDesc("pwh_diff", 0);
  pwh_diff->set_is_dynamic(pwh.is_dynamic());
  pwh_diff->set_data_type(pwh.data_type());

  user_op::TensorDesc* anchors_diff = ctx->MutOutputTensorDesc("anchors_diff", 0);
  anchors_diff->set_is_dynamic(anchors.is_dynamic());
  anchors_diff->set_data_type(anchors.data_type());

  return Maybe<void>::Ok();
}

Maybe<void> FusedGetPboxGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& pxy = ctx->LogicalTensorDesc4InputArgNameAndIndex("pxy", 0);
  FOR_RANGE(int64_t, i, 0, pxy.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("pxy", 0), i)
        .Split(user_op::OpArg("pwh", 0), i)
        .Split(user_op::OpArg("anchors", 0), i)
        .Split(user_op::OpArg("pbox_diff", 0), i)
        .Split(user_op::OpArg("pxy_diff", 0), i)
        .Split(user_op::OpArg("pwh_diff", 0), i)
        .Split(user_op::OpArg("anchors_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
