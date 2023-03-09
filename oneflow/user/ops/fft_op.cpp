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

namespace oneflow{
/* static */ Maybe<void> FftOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
    const Shape& x_shape = ctx->InputShape("x", 0);
    ctx->SetOutputShape("y", 0, x_shape);
    return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FftOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FftOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("other", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  ctx->NewBuilder().PartialSum(user_op::OpArg("x", 0)).PartialSum(user_op::OpArg("other", 0)).PartialSum(user_op::OpArg("y", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftOp::InferDataType(user_op::InferContext* ctx) {
    const DataType& x_data_type = ctx->InputDType("x", 0);
    ctx->SetOutputDType("y", 0, x_data_type);
    return Maybe<void>::Ok();
}


}   // namespace oneflow