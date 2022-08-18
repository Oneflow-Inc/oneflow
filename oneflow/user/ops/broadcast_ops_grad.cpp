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

namespace oneflow {

namespace {

std::string CreateReduceSumLikeBlob(const std::string& in_lbn, const Shape& in_shape,
                                    const std::string& like_lbn, const Shape& like_shape,
                                    const std::string& op_name, const user_op::AddOpFn& AddOp) {
  const Shape& left_extended_shape =
      CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
  if (in_shape == like_shape) {
    return in_lbn;
  } else if (in_shape == left_extended_shape) {
    user_op::UserOpConfWrapperBuilder builder(op_name + "_grad_reshape_like");
    user_op::UserOpConfWrapper grad_op = builder.Op("reshape_like")
                                             .Input("in", in_lbn)
                                             .Input("like", like_lbn)
                                             .Output("out")
                                             .Build();
    AddOp(grad_op);
    return grad_op.output("out", 0);
  } else {
    const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
    user_op::UserOpConfWrapperBuilder builder(op_name + "_grad_reduce_sum_like");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("reduce_sum_like")
            .Input("x", in_lbn)
            .Input("like", like_lbn)
            .Attr<std::vector<int32_t>>("axis",
                                        {broadcast_axis_vec.begin(), broadcast_axis_vec.end()})
            .Output("y")
            .Build();
    AddOp(grad_op);
    return grad_op.output("y", 0);
  }
}

}  // namespace

}  // namespace oneflow
