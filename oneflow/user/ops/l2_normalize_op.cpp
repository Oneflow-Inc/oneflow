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

/* static */ Maybe<void> L2NormalizeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  Shape* y_shape = ctx->MutOutputShape("y", 0);
  Shape* square_x_sum_shape = ctx->MutOutputShape("square_x_sum", 0);
  const int32_t axis = ctx->Attr<int32_t>("axis");
  const float epsilon = ctx->Attr<float>("epsilon");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, x_shape.NumAxes());
  CHECK_GT_OR_RETURN(epsilon, 0);
  *y_shape = x_shape;
  *square_x_sum_shape = x_shape;
  square_x_sum_shape->Set(axis, 1);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> L2NormalizeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> L2NormalizeOp::GetSbp(user_op::SbpContext* ctx) {
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
}

/* static */ Maybe<void> L2NormalizeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("square_x_sum", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> L2NormalizeGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& square_x_sum_shape = ctx->InputShape("square_x_sum", 0);
  Shape* dx_shape = ctx->MutOutputShape("dx", 0);
  const int32_t axis = ctx->Attr<int32_t>("axis");
  const float epsilon = ctx->Attr<float>("epsilon");
  CHECK_EQ_OR_RETURN(dy_shape, y_shape);
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, dy_shape.NumAxes());
  CHECK_GT_OR_RETURN(epsilon, 0);
  FOR_RANGE(int32_t, i, 0, dy_shape.NumAxes()) {
    if (i == axis) {
      CHECK_EQ_OR_RETURN(square_x_sum_shape.At(i), 1);
    } else {
      CHECK_EQ_OR_RETURN(square_x_sum_shape.At(i), dy_shape.At(i));
    }
  }
  *dx_shape = dy_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> L2NormalizeGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> L2NormalizeGradOp::GetSbp(user_op::SbpContext* ctx) {
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
}

/* static */ Maybe<void> L2NormalizeGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("y", 0), ctx->InputDType("dy", 0));
  CHECK_EQ_OR_RETURN(ctx->InputDType("y", 0), ctx->InputDType("square_x_sum", 0));
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("l2_normalize")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
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
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
