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

/*static*/ Maybe<void> Repeat_InterLeaveOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Split(user_op::OpArg("cumsum", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("cumsum", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> Repeat_InterLeaveOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const int64_t repeat_num = ctx->Attr<int64_t>("repeat_num");
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_shape(Shape({repeat_num}));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> Repeat_InterLeaveOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> Repeat_InterLeaveOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
