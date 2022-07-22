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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> CtcLossOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& log_probs = ctx->InputTensorDesc("log_probs", 0);
  const user_op::TensorDesc& targets = ctx->InputTensorDesc("targets", 0);
  const user_op::TensorDesc& input_lengths = ctx->InputTensorDesc("input_lengths", 0);
  const user_op::TensorDesc& target_lengths = ctx->InputTensorDesc("target_lengths", 0);
  const int64_t batch_size = log_probs.shape().At(1);
  const int64_t max_target_length = ctx->Attr<int64_t>("max_target_length");
  if (targets.shape().NumAxes() == 2) {
    CHECK_EQ_OR_RETURN(targets.shape().At(0), batch_size);
    CHECK_GE_OR_RETURN(targets.shape().At(1), max_target_length);
  }
  CHECK_EQ_OR_RETURN(input_lengths.shape().At(0), batch_size);
  CHECK_EQ_OR_RETURN(target_lengths.shape().At(0), batch_size);
  CHECK_GE_OR_RETURN(ctx->Attr<int32_t>("blank"), 0);
  CHECK_LT_OR_RETURN(ctx->Attr<int32_t>("blank"), log_probs.shape().At(2));

  *ctx->MutOutputShape("loss", 0) = Shape({batch_size});
  *ctx->MutOutputShape("alpha", 0) =
      Shape({batch_size, log_probs.shape().At(0), 2 * max_target_length + 1});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CtcLossOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CtcLossOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("log_probs", 0), 1)  // `log_probs` batch axis is 1
      .Split(user_op::OpArg("targets", 0), 0)
      .Split(user_op::OpArg("input_lengths", 0), 0)
      .Split(user_op::OpArg("target_lengths", 0), 0)
      .Split(user_op::OpArg("loss", 0), 0)
      .Split(user_op::OpArg("alpha", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CtcLossOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("loss", 0) = ctx->InputDType("log_probs", 0);
  *ctx->OutputDType("alpha", 0) = ctx->InputDType("log_probs", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CtcLossGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& log_probs = ctx->InputTensorDesc("log_probs", 0);
  const user_op::TensorDesc& targets = ctx->InputTensorDesc("targets", 0);
  const user_op::TensorDesc& input_lengths = ctx->InputTensorDesc("input_lengths", 0);
  const user_op::TensorDesc& target_lengths = ctx->InputTensorDesc("target_lengths", 0);
  const int64_t batch_size = log_probs.shape().At(1);
  const int64_t max_target_length = ctx->Attr<int64_t>("max_target_length");
  if (targets.shape().NumAxes() == 2) {
    CHECK_EQ_OR_RETURN(targets.shape().At(0), batch_size);
    CHECK_GE_OR_RETURN(targets.shape().At(1), max_target_length);
  }
  CHECK_EQ_OR_RETURN(input_lengths.shape().At(0), batch_size);
  CHECK_EQ_OR_RETURN(target_lengths.shape().At(0), batch_size);
  CHECK_GE_OR_RETURN(ctx->Attr<int32_t>("blank"), 0);
  CHECK_LT_OR_RETURN(ctx->Attr<int32_t>("blank"), log_probs.shape().At(2));

  *ctx->MutOutputShape("grad", 0) = log_probs.shape();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CtcLossGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CtcLossGradOp::GetSbp(user_op::SbpContext* ctx) {
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
}

/* static */ Maybe<void> CtcLossGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("grad", 0) = ctx->InputDType("log_probs", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CtcGreedyDecoderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& log_probs = ctx->InputTensorDesc("log_probs", 0);
  const user_op::TensorDesc& input_lengths = ctx->InputTensorDesc("input_lengths", 0);
  const int64_t batch_size = log_probs.shape().At(1);
  CHECK_EQ_OR_RETURN(batch_size, input_lengths.shape().At(0));
  *ctx->MutOutputShape("decoded", 0) = Shape({batch_size, log_probs.shape().At(0)});
  *ctx->MutOutputShape("neg_sum_logits", 0) = Shape({batch_size, 1});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CtcGreedyDecoderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CtcGreedyDecoderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("log_probs", 0), 1)  // `log_probs` batch axis is 1
      .Split(user_op::OpArg("input_lengths", 0), 0)
      .Split(user_op::OpArg("decoded", 0), 0)
      .Split(user_op::OpArg("neg_sum_logits", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CtcGreedyDecoderOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("decoded", 0) = ctx->InputDType("input_lengths", 0);
  *ctx->OutputDType("neg_sum_logits", 0) = ctx->InputDType("log_probs", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("ctc_loss")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
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
            .Attr("max_target_length", ctx->FwOp().attr<int64_t>("max_target_length"))
            .Attr("blank", ctx->FwOp().attr<int32_t>("blank"))
            .Attr("zero_infinity", ctx->FwOp().attr<bool>("zero_infinity"))
            .Output("grad")
            .Build();
      });
      ctx->FwOp().InputGradBind(user_op::OpArg("log_probs", 0),
                                [&ctx, &ctc_loss_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(ctc_loss_grad_op_name).output("grad", 0);
                                });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
