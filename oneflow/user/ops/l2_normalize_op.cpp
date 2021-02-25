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

REGISTER_USER_OP("l2_normalize")
    .Input("x")
    .Output("y")
    .Output("square_x_sum")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* square_x_sum_shape = ctx->Shape4ArgNameAndIndex("square_x_sum", 0);
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const float epsilon = ctx->Attr<float>("epsilon");
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, x_shape->NumAxes());
      CHECK_GT_OR_RETURN(epsilon, 0);
      *y_shape = *x_shape;
      *square_x_sum_shape = *x_shape;
      square_x_sum_shape->Set(axis, 1);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      const int32_t axis = ctx->Attr<int32_t>("axis");
      FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
        if (i != axis) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("x", 0), i)
              .Split(user_op::OpArg("y", 0), i)
              .Split(user_op::OpArg("square_x_sum", 0), i)
              .Build();
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("l2_normalize_grad")
    .Input("dy")
    .Input("y")
    .Input("square_x_sum")
    .Output("dx")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const Shape* square_x_sum_shape = ctx->Shape4ArgNameAndIndex("square_x_sum", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const float epsilon = ctx->Attr<float>("epsilon");
      CHECK_EQ_OR_RETURN(*dy_shape, *y_shape);
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, dy_shape->NumAxes());
      CHECK_GT_OR_RETURN(epsilon, 0);
      FOR_RANGE(int32_t, i, 0, dy_shape->NumAxes()) {
        if (i == axis) {
          CHECK_EQ_OR_RETURN(square_x_sum_shape->At(i), 1);
        } else {
          CHECK_EQ_OR_RETURN(square_x_sum_shape->At(i), dy_shape->At(i));
        }
      }
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      const int32_t axis = ctx->Attr<int32_t>("axis");
      FOR_RANGE(int64_t, i, 0, y_tensor.shape().NumAxes()) {
        if (i != axis) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("y", 0), i)
              .Split(user_op::OpArg("dy", 0), i)
              .Split(user_op::OpArg("square_x_sum", 0), i)
              .Split(user_op::OpArg("dx", 0), i)
              .Build();
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("l2_normalize")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("l2_normalize_grad")
                .Input("y", op.output("y", 0))
                .Input("square_x_sum", op.output("square_x_sum", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("axis", op.attr<int32_t>("axis"))
                .Attr("epsilon", op.attr<float>("epsilon"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
