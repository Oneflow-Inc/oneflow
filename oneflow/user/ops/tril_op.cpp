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

REGISTER_USER_OP("tril")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_fill_value", 0)
    .Attr<int64_t>("integer_fill_value", 0)
    .Attr<bool>("is_floating_fill_value", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in->shape().NumAxes(), 2);
      *out = *in;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      bool fill_zero = ctx->Attr<bool>("is_floating_fill_value")
                           ? (ctx->Attr<double>("floating_fill_value") == static_cast<double>(0))
                           : (ctx->Attr<int64_t>("integer_fill_value") == static_cast<int64_t>(0));
      if (fill_zero) {
        ctx->NewBuilder()
            .PartialSum(user_op::OpArg("in", 0))
            .PartialSum(user_op::OpArg("out", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("tril").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("tril")
                                             .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("out")
                                             .Attr("diagonal", op.attr<int64_t>("diagonal"))
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
});

REGISTER_USER_OP("fused_scale_tril")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("diagonal")
    .Attr<double>("floating_fill_value", 0)
    .Attr<int64_t>("integer_fill_value", 0)
    .Attr<bool>("is_floating_fill_value", false)
    .Attr<double>("floating_scale_value", 1)
    .Attr<int64_t>("integer_scale_value", 1)
    .Attr<bool>("is_floating_scale_value", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in->shape().NumAxes(), 2);
      *out = *in;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      bool fill_zero = ctx->Attr<bool>("is_floating_fill_value")
                           ? (ctx->Attr<double>("floating_fill_value") == static_cast<double>(0))
                           : (ctx->Attr<int64_t>("integer_fill_value") == static_cast<int64_t>(0));
      if (fill_zero) {
        ctx->NewBuilder()
            .PartialSum(user_op::OpArg("in", 0))
            .PartialSum(user_op::OpArg("out", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fused_scale_tril")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fused_scale_tril")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr("diagonal", op.attr<int64_t>("diagonal"))
                .Attr("floating_scale_value", op.attr<double>("floating_scale_value"))
                .Attr("integer_scale_value", op.attr<int64_t>("integer_scale_value"))
                .Attr("is_floating_scale_value", op.attr<bool>("is_floating_scale_value"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
