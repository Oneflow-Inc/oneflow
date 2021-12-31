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

REGISTER_USER_OP("cumsum")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("dim")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_tensor_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      auto dim = ctx->Attr<int64_t>("dim");
      for (auto i = dim + 1; i < in_tensor_desc.shape().NumAxes(); i++) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("cumsum_grad")
    .Input("dy")
    .Output("dx")
    .Attr<int64_t>("dim")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("dx", 0) = ctx->InputShape("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& dy_tensor_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
      for (auto i = 0; i < dy_tensor_desc.shape().NumAxes(); i++) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("dy", 0), i)
            .Split(user_op::OpArg("dx", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("cumsum").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          const user_op::AddOpFn& AddOp)
                                                           -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("cumsum_grad")
                                             .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("dx")
                                             .Attr("dim", op.attr<int64_t>("dim"))
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
    AddOp(grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace

}  // namespace oneflow
