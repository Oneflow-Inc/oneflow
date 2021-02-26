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

REGISTER_USER_OP("combined_margin_loss")
    .Input("x")
    .Input("label")
    .Output("y")
    .Output("theta")
    .Attr<float>("m1")
    .Attr<float>("m2")
    .Attr<float>("m3")
    .Attr<int64_t>("depth")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* label = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      user_op::TensorDesc* theta = ctx->TensorDesc4ArgNameAndIndex("theta", 0);
      CHECK_EQ_OR_RETURN(label->shape().At(0), x->shape().At(0));
      CHECK_GE_OR_RETURN(x->shape().NumAxes(), 2);
      *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *x;
      *theta = *x;
      *theta->mut_shape() = label->shape();
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* label_arg_modifier = GetInputArgModifierFn("label", 0);
      label_arg_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Split(user_op::OpArg("theta", 0), 0)
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 1)
          .Broadcast(user_op::OpArg("label", 0))
          .Split(user_op::OpArg("y", 0), 1)
          .PartialSum(user_op::OpArg("theta", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("combined_margin_loss_grad")
    .Input("dy")
    .Input("label")
    .Input("theta")
    .Output("dx")
    .Attr<float>("m1")
    .Attr<float>("m2")
    .Attr<float>("m3")
    .Attr<int64_t>("depth")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* label = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      const user_op::TensorDesc* theta = ctx->TensorDesc4ArgNameAndIndex("theta", 0);
      CHECK_EQ_OR_RETURN(label->shape().At(0), dy->shape().At(0));
      CHECK_EQ_OR_RETURN(label->shape().At(0), theta->shape().At(0));
      CHECK_GE_OR_RETURN(dy->shape().NumAxes(), 2);
      *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *dy;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("theta", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 1)
          .Broadcast(user_op::OpArg("label", 0))
          .Broadcast(user_op::OpArg("theta", 0))
          .Split(user_op::OpArg("dx", 0), 1)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("combined_margin_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("combined_margin_loss_grad")
                                                 .Input("label", op.input("label", 0))
                                                 .Input("theta", op.output("theta", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("dx")
                                                 .Attr("m1", op.attr<float>("m1"))
                                                 .Attr("m2", op.attr<float>("m2"))
                                                 .Attr("m3", op.attr<float>("m3"))
                                                 .Attr("depth", op.attr<int64_t>("depth"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
