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

REGISTER_USER_OP("hardtanh")
    .Input("in")
    .Attr<double>("min_val")
    .Attr<double>("max_val")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      double min_val = ctx->Attr<double>("min_val");
      double max_val = ctx->Attr<double>("max_val");
      CHECK_LE(min_val, max_val);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("hardtanh_grad")
    .Input("y")
    .Input("dy")
    .Attr<double>("min_val")
    .Attr<double>("max_val")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *y_shape);
      *dx_shape = *dy_shape;
      double min_val = ctx->Attr<double>("min_val");
      double max_val = ctx->Attr<double>("max_val");
      CHECK_LE(min_val, max_val);
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
    });

REGISTER_USER_OP_GRAD("hardtanh").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto hardtanh_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(hardtanh_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("hardtanh_grad")
        .InputBind("y", ctx->FwOp().output("out", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Attr("min_val", ctx->FwOp().attr<double>("min_val"))
        .Attr("max_val", ctx->FwOp().attr<double>("max_val"))
        .Output("dx")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0),
                            [&ctx, &hardtanh_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(hardtanh_grad_op_name).output("dx", 0);
                            });
});

}  // namespace

}  // namespace oneflow
