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

struct DropoutCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float scale;
};

class Dropout : public OpExprGradFunction<DropoutCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DropoutCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const DropoutCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Dropout::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> Dropout::Capture(DropoutCaptureState* state, const TensorTuple& inputs,
                             const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  auto* interp_ctx = dynamic_cast<const DropoutOpInterpCtx*>(ctx);
  state->requires_grad = inputs.at(0)->requires_grad();

  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  state->scale = interp_ctx->scale;
  CHECK_EQ_OR_RETURN(inputs.size(), 2);

  state->SaveTensorForBackward(inputs.at(1));  // mask
  return Maybe<void>::Ok();
}

Maybe<void> Dropout::Apply(const DropoutCaptureState* state, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const std::shared_ptr<oneflow::one::Tensor>& mask = state->SavedTensors().at(0);
  // mask hava no grad(reqiures_grad=False), but still take a place in in_grads
  in_grads->resize(2);
  in_grads->at(0) = JUST(functional::DropoutGrad(out_grads.at(0), mask, state->scale));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("dropout", Dropout);

}  // namespace one
}  // namespace oneflow
