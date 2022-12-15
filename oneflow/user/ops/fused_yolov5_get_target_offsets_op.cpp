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

Maybe<void> FusedYolov5GetTargetOffsetsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& gxy = ctx->InputTensorDesc("gxy", 0);
  const user_op::TensorDesc& gxi = ctx->InputTensorDesc("gxi", 0);

  const Shape& gxy_shape = gxy.shape();

  CHECK_EQ_OR_RETURN(gxy.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(gxy.shape().At(1), 2);
  CHECK_EQ_OR_RETURN(gxy.shape(), gxi.shape());

  user_op::TensorDesc* j = ctx->MutOutputTensorDesc("j", 0);
  j->set_is_dynamic(gxy.is_dynamic());
  j->set_shape(Shape({gxy_shape.At(1) * 2 + 1, gxy_shape.At(0)}));

  return Maybe<void>::Ok();
}

Maybe<void> FusedYolov5GetTargetOffsetsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedYolov5GetTargetOffsetsOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedYolov5GetTargetOffsetsOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& gxy = ctx->InputTensorDesc("gxy", 0);
  const user_op::TensorDesc& gxi = ctx->InputTensorDesc("gxi", 0);

  CHECK_EQ_OR_RETURN(gxy.data_type(), gxi.data_type());

  user_op::TensorDesc* j = ctx->MutOutputTensorDesc("j", 0);
  j->set_data_type(DataType::kBool);
  return Maybe<void>::Ok();
}

Maybe<void> FusedYolov5GetTargetOffsetsOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& gxy = ctx->LogicalTensorDesc4InputArgNameAndIndex("gxy", 0);
  FOR_RANGE(int64_t, i, 0, gxy.shape().NumAxes()) {
    if (i != 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("gxy", 0), i)
          .Split(user_op::OpArg("gxi", 0), i)
          .Split(user_op::OpArg("j", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
