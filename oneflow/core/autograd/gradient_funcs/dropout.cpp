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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct DropoutCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  bool has_addend = false;
  float rate = 0.0;
};

class Dropout : public OpExprGradFunction<DropoutCaptureState> {
 public:
  Maybe<void> Capture(DropoutCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const DropoutCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Dropout::Capture(DropoutCaptureState* state, const TensorTuple& inputs,
                             const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* interp_ctx = dynamic_cast<const DropoutOp*>(ctx);
  state->rate = interp_ctx->rate();
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  if (inputs.size() == 1) {
    state->has_addend = false;
  } else if (inputs.size() == 2) {
    state->has_addend = true;
  } else {
    UNIMPLEMENTED();
  }
  state->SaveTensorForBackward(inputs.at(1));  // mask
  return Maybe<void>::Ok();
}

Maybe<void> Dropout::Apply(const DropoutCaptureState* state, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // Output has y and mask.
  float scale = 0.0f;                       // When dropout rate = 1.0, we set scale as zero.
  if (state->rate < 1.0f) { scale = 1.0f / (1.0f - state->rate); }
  const std::shared_ptr<oneflow::one::Tensor>& mask = state->SavedTensors().at(0);
  if (state->has_addend) {
    in_grads->resize(2);
    in_grads->at(0) = JUST(functional::DropoutGrad(out_grads.at(0), mask, scale));
    in_grads->at(1) = out_grads.at(0);
    return Maybe<void>::Ok();
  } else {
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::DropoutGrad(out_grads.at(0), mask, scale));
    return Maybe<void>::Ok();
  }
}

REGISTER_OP_EXPR_GRAD_FUNCTION("dropout", Dropout);

}  // namespace one
}  // namespace oneflow
