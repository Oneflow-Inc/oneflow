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

REGISTER_USER_OP("multiply")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(x->shape() == y->shape());
      CHECK_OR_RETURN(x->data_type() == y->data_type());
      CHECK_OR_RETURN(x->is_tensor_list() == y->is_tensor_list());
      *out = *x;
      if (x->is_dynamic() || y->is_dynamic()) { *out->mut_is_dynamic() = true; }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("y", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("multiply")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapper x_grad_op =
            user_op::UserOpConfWrapperBuilder(op.op_name() + "_x_grad")
                .Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("out", 0))
                .Input("y", op.input("y", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(x_grad_op.output("out", 0), "x", 0);
        AddOp(x_grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapper y_grad_op =
            user_op::UserOpConfWrapperBuilder(op.op_name() + "_y_grad")
                .Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("out", 0))
                .Input("y", op.input("x", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(y_grad_op.output("out", 0), "y", 0);
        AddOp(y_grad_op);
      }
    });

}  // namespace oneflow
