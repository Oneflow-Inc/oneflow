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
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct DiagCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  int32_t diagonal;
};

class Diag : public OpExprGradFunction<DiagCaptureState> {
 public:
  Maybe<void> Capture(DiagCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const DiagCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Diag::Capture(DiagCaptureState* state, const TensorTuple& inputs,
                          const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  CHECK_EQ_OR_RETURN(outputs.size(), 1);
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  auto* interp_ctx = dynamic_cast<const DiagOpInterpCtx*>(ctx);
  state->diagonal = interp_ctx->diagonal();
  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Diag::Apply(const DiagCaptureState* state, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(2);
  if (state->requires_grad) {
    const auto& x = state->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::DiagGrad(out_grads.at(0), x, state->diagonal));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("diag", Diag);

}  // namespace one
}  // namespace oneflow
