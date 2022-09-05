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

/*static*/ Maybe<void> PackOp::GetSbp(user_op::SbpContext* ctx) {
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
/*static*/ Maybe<void> PackOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  const int32_t pack_num = ctx->Attr<int32_t>("pack_num");
  CHECK_GT_OR_RETURN(pack_num, 0);
  Shape out_shape = in_desc.shape();
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(in_desc.is_dynamic());
  if (out_shape.NumAxes() > 0) {
    out_shape.Set(0, out_shape.At(0) * pack_num);
    out_desc->set_shape(out_shape);
  } else {
    // NOTE(chengcheng): for Scalar input pack
    CHECK_EQ_OR_RETURN(out_shape.elem_cnt(), 1);
    out_desc->set_shape(Shape({pack_num}));
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> PackOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return PackOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> PackOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PackOp::InferOutputBlobTimeShape(
    user_op::InferOutputBlobTimeShapeFnContext* ctx) {
  const int32_t pack_num = ctx->user_op_conf().attr<int32_t>("pack_num");
  DimVector time_shape_dim_vec = ctx->TimeShape4InputArgNameAndIndex("in", 0).dim_vec();
  CHECK_OR_RETURN(!time_shape_dim_vec.empty());
  CHECK_EQ_OR_RETURN(time_shape_dim_vec.back(), pack_num);
  time_shape_dim_vec.pop_back();
  if (time_shape_dim_vec.empty()) { time_shape_dim_vec.emplace_back(1); }
  *ctx->mut_output_blob_time_shape() = Shape(time_shape_dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
