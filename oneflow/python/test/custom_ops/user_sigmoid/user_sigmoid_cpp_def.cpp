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

REGISTER_USER_OP("user_sigmoid_forward")
    .Input("x")
    .Output("y")
    .Attr<std::string>("device_sub_tag", "py")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      *out_shape = *in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("user_sigmoid_backward")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr<std::string>("device_sub_tag", "py")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *y_shape);
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      FOR_RANGE(int64_t, i, 0, y_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("y", 0), i)
            .Split(user_op::OpArg("dy", 0), i)
            .Split(user_op::OpArg("dx", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("y", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("user_sigmoid_forward")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("user_sigmoid_backward")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      };
      ctx->DefineOp(grad_op_name, grad_op_func);

      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
    });

}  // namespace

}  // namespace oneflow
