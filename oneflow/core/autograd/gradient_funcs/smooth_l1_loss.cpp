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

struct SmoothL1LossCaptureState : public AutoGradCaptureState {
  std::string reduction = "none";
  float beta = 0.0;
  bool requires_grad = false;
};

class SmoothL1Loss : public OpExprGradFunction<SmoothL1LossCaptureState> {
 public:
  Maybe<void> Capture(SmoothL1LossCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->requires_grad = inputs.at(0)->requires_grad();  // prediction
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    state->SaveTensorForBackward(inputs.at(0));  // prediction
    state->SaveTensorForBackward(inputs.at(1));  // label

    auto* interp_ctx = dynamic_cast<const SmoothL1LossOpInterpCtx*>(ctx);
    state->beta = interp_ctx->beta;
    state->reduction = interp_ctx->reduction;
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SmoothL1LossCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(2);

    if (state->requires_grad) {
      const auto& prediction = state->SavedTensors().at(0);
      const auto& label = state->SavedTensors().at(1);
      in_grads->at(0) = JUST(functional::SmoothL1LossGrad(out_grads.at(0), prediction, label,
                                                          state->beta, state->reduction));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("smooth_l1_loss", SmoothL1Loss);  // todo: name

}  // namespace one
}  // namespace oneflow
