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

REGISTER_USER_OP("softmax")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), axis)
            .Split(user_op::OpArg("out", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("softmax_grad")
    .Input("y")
    .Input("dy")
    .Output("dx")
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
      FOR_RANGE(int64_t, axis, 0, y_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("y", 0), axis)
            .Split(user_op::OpArg("dy", 0), axis)
            .Split(user_op::OpArg("dx", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("softmax").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper softmax_grad_op =
        builder.Op("softmax_grad")
            .Input("y", op.output("out", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx")
            .Build();
    op.BindGradTensorWithOpInput(softmax_grad_op.output("dx", 0), "in", 0);
    AddOp(softmax_grad_op);
  }
});

REGISTER_USER_OP("fused_tril_scale_softmax_mask_and_scale")
    .Input("in")
    .Input("mask")
    .Output("out")
    .Output("softmax_out")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_fill_value", 0)
    .Attr<int64_t>("integer_fill_value", 0)
    .Attr<bool>("is_floating_fill_value", false)
    .Attr<double>("floating_scale_value", 1)
    .Attr<int64_t>("integer_scale_value", 1)
    .Attr<bool>("is_floating_scale_value", false)
    .Attr<float>("scale")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *ctx->TensorDesc4ArgNameAndIndex("softmax_out", 0) =
          *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
      CHECK(mask_modifier != nullptr);
      mask_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("out", 0), axis)
            .Split(user_op::OpArg("softmax_out", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("fused_tril_scale_softmax_mask_and_scale_grad")
    .Input("y")
    .Input("dy")
    .Input("mask")
    .Output("dx")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_scale_value", 1)
    .Attr<int64_t>("integer_scale_value", 1)
    .Attr<bool>("is_floating_scale_value", false)
    .Attr<float>("scale")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      CHECK(*dy_desc == *y_desc);
      *dx_desc = *dy_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      FOR_RANGE(int64_t, axis, 0, y_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("y", 0), axis)
            .Split(user_op::OpArg("dy", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("dx", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fused_tril_scale_softmax_mask_and_scale")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fused_tril_scale_softmax_mask_and_scale_grad")
                .Input("y", op.output("softmax_out", 0))
                .Input("mask", op.input("mask", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("dx")
                .Attr("diagonal", op.attr<int64_t>("diagonal"))
                .Attr("floating_scale_value", op.attr<double>("floating_scale_value"))
                .Attr("integer_scale_value", op.attr<int64_t>("integer_scale_value"))
                .Attr("is_floating_scale_value", op.attr<bool>("is_floating_scale_value"))
                .Attr("scale", op.attr<float>("scale"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "in", 0);
        AddOp(grad_op);
      }
    });

}  // namespace

}  // namespace oneflow
