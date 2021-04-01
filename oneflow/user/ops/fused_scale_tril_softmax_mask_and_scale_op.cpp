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

REGISTER_USER_OP("fused_tril_scale_softmax_mask_and_scale")
    .Input("x")
    .Input("mask")
    .Output("y")
    .Output("softmax_y")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_tril_fill_value", 0)
    .Attr<int64_t>("integer_tril_fill_value", 0)
    .Attr<bool>("is_floating_tril_fill_value", false)
    .Attr<double>("floating_prologue_scale_value", 1)
    .Attr<int64_t>("integer_prologue_scale_value", 1)
    .Attr<bool>("is_floating_prologue_scale_value", false)
    .Attr<float>("epilogue_scale_value")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *x_desc;
      *ctx->TensorDesc4ArgNameAndIndex("softmax_y", 0) = *x_desc;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
      CHECK(mask_modifier != nullptr);
      mask_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      CHECK_GE(x_tensor.shape().NumAxes(), 2);
      FOR_RANGE(int64_t, axis, 0, x_tensor.shape().NumAxes() - 2) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("x", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("y", 0), axis)
            .Split(user_op::OpArg("softmax_y", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("fused_tril_scale_softmax_mask_and_scale_grad")
    .Input("softmax_y")
    .Input("dy")
    .Input("mask")
    .Output("dx")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_epilogue_scale_value", 1)
    .Attr<int64_t>("integer_epilogue_scale_value", 1)
    .Attr<bool>("is_floating_epilogue_scale_value", false)
    .Attr<float>("prologue_scale_value")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* softmax_y_desc = ctx->TensorDesc4ArgNameAndIndex("softmax_y", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      CHECK(dy_desc->shape() == softmax_y_desc->shape());
      CHECK(dy_desc->data_type() == softmax_y_desc->data_type());
      *dx_desc = *dy_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      CHECK_GE(y_tensor.shape().NumAxes(), 2);
      FOR_RANGE(int64_t, axis, 0, y_tensor.shape().NumAxes() - 2) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("softmax_y", 0), axis)
            .Split(user_op::OpArg("dy", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("dx", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fused_tril_scale_softmax_mask_and_scale")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fused_tril_scale_softmax_mask_and_scale_grad")
                .Input("softmax_y", op.output("softmax_y", 0))
                .Input("mask", op.input("mask", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("prologue_scale_value", op.attr<float>("epilogue_scale_value"))
                .Attr("diagonal", op.attr<int64_t>("diagonal"))
                .Attr("floating_epilogue_scale_value",
                      op.attr<double>("floating_prologue_scale_value"))
                .Attr("integer_epilogue_scale_value",
                      op.attr<int64_t>("integer_prologue_scale_value"))
                .Attr("is_floating_epilogue_scale_value",
                      op.attr<bool>("is_floating_prologue_scale_value"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace

}  // namespace oneflow
