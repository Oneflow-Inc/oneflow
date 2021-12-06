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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct KLDivLossCaptureState : public AutoGradCaptureState {
  bool log_target = false;
  std::string reduction = "";
};

class KLDivLoss : public OpExprGradFunction<KLDivLossCaptureState> {
 public:
  Maybe<void> Capture(KLDivLossCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const KLDivLossCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> KLDivLoss::Capture(KLDivLossCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  auto* interp_ctx = dynamic_cast<const KlDivLossOpInterpCtx*>(ctx);
  state->log_target = interp_ctx->log_target;
  state->reduction = interp_ctx->reduction;
  state->SaveTensorForBackward(inputs.at(0));  // input
  state->SaveTensorForBackward(inputs.at(1));  // target
  return Maybe<void>::Ok();
}
Maybe<void> KLDivLoss::Apply(const KLDivLossCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);
  const auto& input = state->SavedTensors().at(0);
  const auto& target = state->SavedTensors().at(1);
  in_grads->resize(state->SavedTensors().size());
  in_grads->at(0) =
      JUST(functional::KLDivLossGrad(dy, input, target, state->log_target, state->reduction));

  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("kl_div_loss", KLDivLoss);

}  // namespace one
}  // namespace oneflow
