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

namespace {

int32_t TransformNegativeAxisToPositive(int32_t axis, const int32_t num_axes) {
  axis = axis < 0 ? axis + num_axes + 1 : axis;
  CHECK_GE(axis, 0);
  CHECK_LE(axis, num_axes);
  return axis;
}

}  // namespace

/* static */ Maybe<void> ExpandDimsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const int32_t axis =
      TransformNegativeAxisToPositive(ctx->Attr<int32_t>("axis"), in_shape.NumAxes());

  auto dim_vec = in_shape.dim_vec();
  dim_vec.insert(dim_vec.begin() + axis, 1);
  ctx->SetOutputShape("out", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ExpandDimsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ExpandDimsOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const int32_t axis =
      TransformNegativeAxisToPositive(ctx->Attr<int32_t>("axis"), in_tensor.shape().NumAxes());

  auto dim_vec = in_tensor.shape().dim_vec();
  FOR_RANGE(int32_t, in_axis, 0, dim_vec.size()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), in_axis)
        .Split(user_op::OpArg("out", 0), in_axis < axis ? in_axis : in_axis + 1)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandDimsOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
