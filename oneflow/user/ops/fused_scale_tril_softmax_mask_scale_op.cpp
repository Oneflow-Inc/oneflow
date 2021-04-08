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

REGISTER_USER_OP("fused_tril_scale_softmax_mask_scale")
    .Input("x")
    .Input("mask")
    .Output("y")
    .Output("softmax_y")
    .Attr<int64_t>("diagonal")
    .Attr<float>("tril_fill_value", 0)
    .Attr<float>("tril_scale_value", 1.0)
    .Attr<float>("mask_scale_value", 1.0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      *ctx->Shape4ArgNameAndIndex("y", 0) = x_desc->shape();
      *ctx->IsDynamic4ArgNameAndIndex("y", 0) = x_desc->is_dynamic();
      *ctx->Shape4ArgNameAndIndex("softmax_y", 0) = x_desc->shape();
      *ctx->IsDynamic4ArgNameAndIndex("softmax_y", 0) = x_desc->is_dynamic();
      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      *ctx->Dtype4ArgNameAndIndex("y", 0) = x_desc->data_type();
      *ctx->Dtype4ArgNameAndIndex("softmax_y", 0) = x_desc->data_type();
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

REGISTER_USER_OP("fused_tril_scale_softmax_mask_scale_grad")
    .Input("softmax_y")
    .Input("dy")
    .Input("mask")
    .Output("dx")
    .Attr<int64_t>("diagonal")
    .Attr<float>("tril_scale_value")
    .Attr<float>("mask_scale_value")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* softmax_y_desc = ctx->TensorDesc4ArgNameAndIndex("softmax_y", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      CHECK(dy_desc->shape() == softmax_y_desc->shape());
      *dx_desc->mut_shape() = dy_desc->shape();
      *dx_desc->mut_is_dynamic() = dy_desc->is_dynamic();
      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* softmax_y_desc = ctx->TensorDesc4ArgNameAndIndex("softmax_y", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      CHECK(dy_desc->data_type() == softmax_y_desc->data_type());
      *dx_desc->mut_data_type() = dy_desc->data_type();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
      CHECK_GE(dy_tensor.shape().NumAxes(), 2);
      FOR_RANGE(int64_t, axis, 0, dy_tensor.shape().NumAxes() - 2) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("softmax_y", 0), axis)
            .Split(user_op::OpArg("dy", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("dx", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fused_tril_scale_softmax_mask_scale")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fused_tril_scale_softmax_mask_scale_grad")
                .Input("softmax_y", op.output("softmax_y", 0))
                .Input("mask", op.input("mask", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("diagonal", op.attr<int64_t>("diagonal"))
                .Attr("tril_scale_value", op.attr<float>("tril_scale_value"))
                .Attr("mask_scale_value", op.attr<float>("mask_scale_value"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace

}  // namespace oneflow
