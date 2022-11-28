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

/*static*/ Maybe<void> UnpackOp::GetSbp(user_op::SbpContext* ctx) {
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
/*static*/ Maybe<void> UnpackOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  const Shape& in_shape = in_desc.shape();
  CHECK_GT_OR_RETURN(in_shape.NumAxes(), 0);
  const auto unpack_num = ctx->Attr<int32_t>("unpack_num");
  CHECK_EQ_OR_RETURN(in_shape.At(0) % unpack_num, 0);
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  Shape out_shape = in_desc.shape();
  out_shape.Set(0, in_shape.At(0) / unpack_num);
  out_desc->set_shape(out_shape);
  out_desc->set_is_dynamic(in_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnpackOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UnpackOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  out_desc->set_data_type(in_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnpackOp::InferOutputBlobTimeShape(
    user_op::InferOutputBlobTimeShapeFnContext* ctx) {
  const int32_t unpack_num = ctx->user_op_conf().attr<int32_t>("unpack_num");
  DimVector time_shape_dim_vec = ctx->TimeShape4InputArgNameAndIndex("in", 0).dim_vec();
  time_shape_dim_vec.emplace_back(unpack_num);
  *ctx->mut_output_blob_time_shape() = Shape(time_shape_dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
