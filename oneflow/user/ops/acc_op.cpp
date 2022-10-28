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

/*static*/ Maybe<void> AccOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> AccOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> AccOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return AccOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> AccOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> AccOp::InferOutputBlobTimeShape(
    user_op::InferOutputBlobTimeShapeFnContext* ctx) {
  const int32_t max_acc_num = ctx->user_op_conf().attr<int32_t>("max_acc_num");
  const Shape& in_time_shape = ctx->TimeShape4InputArgNameAndIndex("in", 0);
  DimVector time_shape_dim_vec = in_time_shape.dim_vec();
  CHECK_OR_RETURN(!time_shape_dim_vec.empty());
  if (time_shape_dim_vec.back() == max_acc_num) {
    time_shape_dim_vec.pop_back();
  } else if (time_shape_dim_vec.back() % max_acc_num == 0) {
    time_shape_dim_vec.back() /= max_acc_num;
  } else {
    const int64_t elem_cnt = in_time_shape.elem_cnt();
    time_shape_dim_vec.resize(1);
    time_shape_dim_vec.back() = elem_cnt / max_acc_num;
  }
  *ctx->mut_output_blob_time_shape() = Shape(time_shape_dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
