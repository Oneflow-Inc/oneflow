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

Maybe<void> TransformNegativeAxesToPositive(const std::vector<int32_t>& axes_vec,
                                            const int32_t num_axes, AxisVector* fixed_axes_vec) {
  fixed_axes_vec->resize(axes_vec.size());
  FOR_RANGE(size_t, i, 0, fixed_axes_vec->size()) {
    CHECK_GE(axes_vec[i], -num_axes);
    CHECK_LT(axes_vec[i], num_axes);
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

REGISTER_USER_OP("squeeze")
    .Input("in")
    .Output("out")
    .Attr<std::vector<int32_t>>("axes")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      AxisVector fixed_axes_vec;
      TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"), in_shape->NumAxes(),
                                      &fixed_axes_vec);

      DimVector dim_vec = in_shape->dim_vec();
      CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &dim_vec);
      dim_vec.erase(std::remove(dim_vec.begin(), dim_vec.end(), -1), dim_vec.end());
      if (dim_vec.empty()) {
        *out_shape = Shape({1});
      } else {
        *out_shape = Shape(dim_vec);
      }
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      AxisVector fixed_axes_vec;
      TransformNegativeAxesToPositive(ctx->Attr<std::vector<int32_t>>("axes"),
                                      in_tensor.shape().NumAxes(), &fixed_axes_vec);

      DimVector dim_vec = in_tensor.shape().dim_vec();
      CheckAndLabelAxesToSqueezeMinusOne(fixed_axes_vec, &dim_vec);
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
    });

REGISTER_USER_OP_GRAD("squeeze").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
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
});

}  // namespace oneflow
