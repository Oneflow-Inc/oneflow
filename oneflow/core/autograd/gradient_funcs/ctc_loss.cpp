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
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct CTCLossCaptureState : public AutoGradCaptureState {
  int64_t max_target_length;
  int32_t blank;
  bool zero_infinity;
  bool requires_grad;
};

class CTCLoss : public OpExprGradFunction<CTCLossCaptureState> {
 public:
  Maybe<void> Capture(CTCLossCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const CTCLossCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> CTCLoss::Capture(CTCLossCaptureState* state, const TensorTuple& inputs,
                             const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* interp_ctx = dynamic_cast<const CtcLossOp*>(ctx);
  state->max_target_length = interp_ctx->max_target_length();
  state->blank = interp_ctx->blank();
  state->zero_infinity = interp_ctx->zero_infinity();

  CHECK_EQ_OR_RETURN(inputs.size(), 4);
  CHECK_EQ_OR_RETURN(outputs.size(), 2);
  state->SaveTensorForBackward(outputs.at(0));  // loss
  state->SaveTensorForBackward(outputs.at(1));  // alpha
  state->SaveTensorForBackward(inputs.at(0));   // log_probs
  state->SaveTensorForBackward(inputs.at(1));   // targets
  state->SaveTensorForBackward(inputs.at(2));   // input_lengths
  state->SaveTensorForBackward(inputs.at(3));   // target_lengths
  return Maybe<void>::Ok();
}

Maybe<void> CTCLoss::Apply(const CTCLossCaptureState* state, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);

  const auto& grad_out = out_grads.at(0);
  const auto& loss = state->SavedTensors().at(0);
  const auto& alpha = state->SavedTensors().at(1);
  const auto& log_probs = state->SavedTensors().at(2);
  const auto& targets = state->SavedTensors().at(3);
  const auto& input_lengths = state->SavedTensors().at(4);
  const auto& target_lengths = state->SavedTensors().at(5);
  in_grads->resize(4);
  in_grads->at(0) = JUST(functional::CtcLossGrad(grad_out, log_probs, targets, input_lengths,
                                                 target_lengths, loss, alpha, state->blank,
                                                 state->zero_infinity, state->max_target_length));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("ctc_loss", CTCLoss);

}  // namespace one
}  // namespace oneflow
