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

Maybe<void> CheckAndLabelAxesToSqueezeMinusOne(const AxisVector& axes, Shape* shape) {
  for (const auto& axis : axes) {
    CHECK_EQ_OR_RETURN(shape->at(axis), 1);
    shape->at(axis) = -1;
  }
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> SqueezeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  AxisVector fixed_axes_vec;
  JUST(TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"),
                                       in_tensor.shape().NumAxes(), &fixed_axes_vec));

  Shape shape = in_tensor.shape();
  JUST(CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &shape));
  int32_t out_axis = 0;
  FOR_RANGE(int32_t, in_axis, 0, shape.size()) {
    if (shape.at(in_axis) != -1) {
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
  Shape* out_shape = ctx->OutputShape("out", 0);
  AxisVector fixed_axes_vec;
  JUST(TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"), in_shape.NumAxes(),
                                       &fixed_axes_vec));

  Shape shape = in_shape;
  JUST(CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &shape));
  shape.erase(std::remove(shape.begin(), shape.end(), -1), shape.end());
  *out_shape = shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SqueezeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SqueezeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("squeeze").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("reshape_like")
                                             .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Input("like", op.input("in", 0))
                                             .Output("out")
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace oneflow
