
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

REGISTER_USER_OP("py_op")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // TODO(strint):out tensor infer
      const auto* in_0 = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      auto* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_NOTNULL_OR_RETURN(in_0);
      CHECK_NOTNULL_OR_RETURN(out);
      *out = *in_0;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      return user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis(ctx);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      for (int64_t i = 0; i < num_axes; ++i) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("out", 0), i).Build();
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(user_op::OpArg("out", 0)).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("py_op").SetBackwardOpConGenfFn([](user_op::BackwardOpConfContext* ctx) {
  const auto py_op_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(py_op_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("py_op_grad")
        .InputBind("x", ctx->FwOp().input("in", 0))
        .InputBind("y", ctx->FwOp().output("out", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Output("dx")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0),
                            [&ctx, &py_op_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(py_op_grad_op_name).output("dx", 0);
                            });
});

}  // namespace oneflow
