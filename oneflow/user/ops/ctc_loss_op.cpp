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

REGISTER_USER_OP("ctc_loss")
    .Input("log_probs")
    .Input("targets")
    .Input("input_lengths")
    .Input("target_lengths")
    .Output("loss")
    .Output("alpha")  // 'alpha' is just for compute log_probs's grad, alpha's grad will be ignored
    .Attr<int>("blank")
    .Attr<bool>("zero_infinity")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* log_probs = ctx->TensorDesc4ArgNameAndIndex("log_probs", 0);
      const user_op::TensorDesc* targets = ctx->TensorDesc4ArgNameAndIndex("targets", 0);
      const user_op::TensorDesc* input_lengths =
          ctx->TensorDesc4ArgNameAndIndex("input_lengths", 0);
      const user_op::TensorDesc* target_lengths =
          ctx->TensorDesc4ArgNameAndIndex("target_lengths", 0);
      const int64_t batch_size = log_probs->shape().At(1);
      CHECK_EQ_OR_RETURN(batch_size, targets->shape().At(0));
      CHECK_EQ_OR_RETURN(batch_size, input_lengths->shape().At(0));
      CHECK_EQ_OR_RETURN(batch_size, target_lengths->shape().At(0));
      CHECK_GE_OR_RETURN(ctx->Attr<int>("blank"), 0);
      *ctx->Dtype4ArgNameAndIndex("loss", 0) = *ctx->Dtype4ArgNameAndIndex("log_probs", 0);
      *ctx->Shape4ArgNameAndIndex("loss", 0) = Shape({batch_size});
      *ctx->Dtype4ArgNameAndIndex("alpha", 0) = *ctx->Dtype4ArgNameAndIndex("log_probs", 0);
      *ctx->Shape4ArgNameAndIndex("alpha", 0) =
          Shape({batch_size, log_probs->shape().At(0), 2 * targets->shape().At(1) + 1});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("log_probs", 0), 1)  // `log_probs` batch axis is 1
          .Split(user_op::OpArg("targets", 0), 0)
          .Split(user_op::OpArg("input_lengths", 0), 0)
          .Split(user_op::OpArg("target_lengths", 0), 0)
          .Split(user_op::OpArg("loss", 0), 0)
          .Split(user_op::OpArg("alpha", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("ctc_loss_grad")
    .Input("grad_out")
    .Input("log_probs")
    .Input("targets")
    .Input("input_lengths")
    .Input("target_lengths")
    .Input("loss")
    .Input("alpha")
    .Output("grad")
    .Attr<int>("blank")
    .Attr<bool>("zero_infinity")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* log_probs = ctx->TensorDesc4ArgNameAndIndex("log_probs", 0);
      const user_op::TensorDesc* targets = ctx->TensorDesc4ArgNameAndIndex("targets", 0);
      const user_op::TensorDesc* input_lengths =
          ctx->TensorDesc4ArgNameAndIndex("input_lengths", 0);
      const user_op::TensorDesc* target_lengths =
          ctx->TensorDesc4ArgNameAndIndex("target_lengths", 0);
      const int64_t batch_size = log_probs->shape().At(1);
      CHECK_EQ_OR_RETURN(batch_size, targets->shape().At(0));
      CHECK_EQ_OR_RETURN(batch_size, input_lengths->shape().At(0));
      CHECK_EQ_OR_RETURN(batch_size, target_lengths->shape().At(0));
      CHECK_GE_OR_RETURN(ctx->Attr<int>("blank"), 0);
      *ctx->Dtype4ArgNameAndIndex("grad", 0) = *ctx->Dtype4ArgNameAndIndex("log_probs", 0);
      *ctx->Shape4ArgNameAndIndex("grad", 0) = log_probs->shape();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("grad_out", 0), 0)
          .Split(user_op::OpArg("log_probs", 0), 1)  // `log_probs` batch axis is 1
          .Split(user_op::OpArg("targets", 0), 0)
          .Split(user_op::OpArg("input_lengths", 0), 0)
          .Split(user_op::OpArg("target_lengths", 0), 0)
          .Split(user_op::OpArg("loss", 0), 0)
          .Split(user_op::OpArg("alpha", 0), 0)
          .Split(user_op::OpArg("grad", 0), 1)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("ctc_loss").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto ctc_loss_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(ctc_loss_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("ctc_loss_grad")
        .InputBind("grad_out", ctx->FwOp().output_grad("loss", 0))
        .InputBind("log_probs", ctx->FwOp().input("log_probs", 0))
        .InputBind("targets", ctx->FwOp().input("targets", 0))
        .InputBind("input_lengths", ctx->FwOp().input("input_lengths", 0))
        .InputBind("target_lengths", ctx->FwOp().input("target_lengths", 0))
        .InputBind("loss", ctx->FwOp().output("loss", 0))
        .InputBind("alpha", ctx->FwOp().output("alpha", 0))
        .Attr("blank", ctx->FwOp().attr<int>("blank"))
        .Attr("zero_infinity", ctx->FwOp().attr<bool>("zero_infinity"))
        .Output("grad")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("log_probs", 0),
                            [&ctx, &ctc_loss_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(ctc_loss_grad_op_name).output("grad", 0);
                            });
});

}  // namespace oneflow
