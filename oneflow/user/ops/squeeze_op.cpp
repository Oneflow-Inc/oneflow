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

Maybe<void> TransformNegativeAxesToPositive(const std::vector<int32_t>& axes_vec,
                                            const int32_t num_axes, AxisVector* fixed_axes_vec) {
  fixed_axes_vec->resize(axes_vec.size());
  FOR_RANGE(size_t, i, 0, fixed_axes_vec->size()) {
    CHECK_GE_OR_RETURN(axes_vec[i], -num_axes);
    CHECK_LT_OR_RETURN(axes_vec[i], num_axes);
    fixed_axes_vec->at(i) = axes_vec[i] >= 0 ? axes_vec[i] : axes_vec[i] + num_axes;
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckAndLabelAxesToSqueezeMinusOne(const AxisVector& axes, DimVector* dim_vec) {
  for (const auto& axis : axes) {
    CHECK_EQ_OR_RETURN(dim_vec->at(axis), 1);
    dim_vec->at(axis) = -1;
  }
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> SqueezeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  AxisVector fixed_axes_vec;
  JUST(TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"),
                                       in_tensor.shape().NumAxes(), &fixed_axes_vec));

  DimVector dim_vec = in_tensor.shape().dim_vec();
  JUST(CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &dim_vec));
  int32_t out_axis = 0;
  FOR_RANGE(int32_t, in_axis, 0, dim_vec.size()) {
    if (dim_vec.at(in_axis) != -1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), in_axis)
          .Split(user_op::OpArg("out", 0), out_axis)
          .Build();
      ++out_axis;
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SqueezeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  AxisVector fixed_axes_vec;
  JUST(TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"), in_shape.NumAxes(),
                                       &fixed_axes_vec));

  DimVector dim_vec = in_shape.dim_vec();
  JUST(CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &dim_vec));
  dim_vec.erase(std::remove(dim_vec.begin(), dim_vec.end(), -1), dim_vec.end());
  ctx->SetOutputShape("out", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SqueezeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SqueezeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
