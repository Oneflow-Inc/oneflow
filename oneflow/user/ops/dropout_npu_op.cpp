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

/* static */ Maybe<void> DropoutNpuOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  ctx->SetOutputShape("out", 0, in_shape);
  uint32_t numels = 1;
  for(auto size:in_shape)
  {
    numels *= size;
  }
  uint32_t length = (numels + 128 - 1) / 128 * 128;
  DimVector indice_dim = {length / 8};
  Shape new_shape(indice_dim);
  ctx->SetOutputShape("mask", 0, new_shape);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DropoutNpuOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DropoutNpuOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutNpuOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                              const user_op::UserOpConfWrapper& conf) {
  float rate = conf.attr<float>("rate");
  CHECK_GE_OR_RETURN(rate, 0.0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DropoutNpuOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("mask", 0, DataType::kUInt8);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
