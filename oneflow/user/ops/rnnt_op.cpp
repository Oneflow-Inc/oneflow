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

REGISTER_USER_OP("RNNTloss")
    .Input("acts")
    .Input("labels")
    .Input("act_lens")
    .Input("label_lens")
    .Attr<int32_t>("blank_label", 0)
    .Attr<int32_t>("num_threads", 0)
    .Output("costs") 
    .Output("grads")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& acts = ctx->InputTensorDesc("acts", 0);
      const ShapeView& acts_shape = acts.shape();

      DimVector costs_dim_vec = {acts_shape.At(0)};
      user_op::TensorDesc* costs_desc = ctx->OutputTensorDesc("costs", 0);
      costs_desc->set_is_dynamic(false);
      *costs_desc->mut_shape() = Shape(costs_dim_vec);

      *ctx->OutputShape("grads", 0) = ctx->InputShape("acts", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("costs", 0) = ctx->InputDType("acts", 0);
      *ctx->OutputDType("grads", 0) = ctx->InputDType("acts", 0);
      return Maybe<void>::Ok();
    });


REGISTER_USER_OP_GRAD("RNNTloss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("acts", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper rnntloss_grad_op =
            builder.Op("RNNTloss")
                .Input("grads", op.output("grads", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("costs", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(rnntloss_grad_op.output("dx", 0), "acts", 0);
        AddOp(rnntloss_grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace
}  // namespace oneflow
