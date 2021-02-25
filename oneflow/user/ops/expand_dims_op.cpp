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

int32_t TransformNegativeAxisToPositive(int32_t axis, const int32_t num_axes) {
  axis = axis < 0 ? axis + num_axes + 1 : axis;
  CHECK_GE(axis, 0);
  CHECK_LE(axis, num_axes);
  return axis;
}

}  // namespace

REGISTER_USER_OP("expand_dims")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("axis")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const int32_t axis =
          TransformNegativeAxisToPositive(ctx->Attr<int32_t>("axis"), in_shape->NumAxes());

      auto dim_vec = in_shape->dim_vec();
      dim_vec.insert(dim_vec.begin() + axis, 1);
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    });

REGISTER_USER_OP_GRAD("expand_dims")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("reshape_like")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Input("like", op.input("in", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
