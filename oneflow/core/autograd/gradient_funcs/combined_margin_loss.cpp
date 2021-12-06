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

struct CombinedMarginLossCaptureState : public AutoGradCaptureState {
  float m1;
  float m2;
  float m3;
  int64_t depth;
  size_t label_index;
  size_t theta_index;
  bool requires_grad;
};

class CombinedMarginLoss : public OpExprGradFunction<CombinedMarginLossCaptureState> {
 public:
  Maybe<void> Capture(CombinedMarginLossCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->requires_grad = inputs.at(0)->requires_grad();  // x
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    state->label_index = state->SaveTensorForBackward(inputs.at(1));   // label
    state->theta_index = state->SaveTensorForBackward(outputs.at(1));  // theta

    auto* interp_ctx = dynamic_cast<const CombinedMarginLossOpInterpCtx*>(ctx);
    state->m1 = interp_ctx->m1;
    state->m2 = interp_ctx->m2;
    state->m3 = interp_ctx->m3;
    state->depth = interp_ctx->depth;
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CombinedMarginLossCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);
    in_grads->resize(2);

    if (state->requires_grad) {
      const auto& label = state->SavedTensors().at(state->label_index);
      const auto& theta = state->SavedTensors().at(state->theta_index);
      in_grads->at(0) = JUST(functional::CombinedMarginLossGrad(
          out_grads.at(0), label, theta, state->m1, state->m2, state->m3, state->depth));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("combined_margin_loss", CombinedMarginLoss);

}  // namespace one
}  // namespace oneflow
