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

/*static*/ Maybe<void> TfPreluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const user_op::TensorDesc& alpha_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("alpha", 0))
      .Split(user_op::OpArg("y", 0), 0)
      .Build();
  FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
    if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("alpha", 0), i - 1)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  const Shape& alpha_shape = ctx->InputShape("alpha", 0);
  CHECK_EQ_OR_RETURN(x_desc.shape().NumAxes(), alpha_shape.NumAxes() + 1);
  FOR_RANGE(int64_t, i, 1, x_desc.shape().NumAxes()) {
    CHECK_OR_RETURN((alpha_shape.At(i - 1) == x_desc.shape().At(i))
                    || (alpha_shape.At(i - 1) == 1));
  }
  y_desc->set_shape(x_desc.shape());
  y_desc->set_is_dynamic(x_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TfPreluOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TfPreluGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const user_op::TensorDesc& alpha_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("alpha", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("alpha", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
    if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), i)
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("alpha", 0), i - 1)
          .Split(user_op::OpArg("dx", 0), i)
          .Split(user_op::OpArg("alpha_diff", 0), i - 1)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  const user_op::TensorDesc& alpha_desc = ctx->InputTensorDesc("alpha", 0);
  CHECK_EQ_OR_RETURN(x_desc.shape().NumAxes(), alpha_desc.shape().NumAxes() + 1);
  FOR_RANGE(int64_t, i, 1, x_desc.shape().NumAxes()) {
    CHECK_OR_RETURN((alpha_desc.shape().At(i - 1) == x_desc.shape().At(i))
                    || (alpha_desc.shape().At(i - 1) == 1));
  }
  CHECK_EQ_OR_RETURN(dy_desc.shape(), x_desc.shape());
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), x_desc.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("dy", 0))
      << ", but got " << DataType_Name(x_desc.data_type());
  dx_desc->set_shape(x_desc.shape());
  dx_desc->set_is_dynamic(x_desc.is_dynamic());
  ctx->SetOutputShape("alpha_diff", 0, alpha_desc.shape());
  ctx->SetOutputIsDynamic("alpha_diff", 0, alpha_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TfPreluGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  ctx->SetOutputDType("alpha_diff", 0, ctx->InputDType("alpha", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
