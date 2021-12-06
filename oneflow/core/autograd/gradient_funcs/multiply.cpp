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

struct MultiplyCaptureState : public AutoGradCaptureState {
  bool requires_grad_x;
  bool requires_grad_y;
  int32_t index_x;
  int32_t index_y;
};

class Multiply : public OpExprGradFunction<MultiplyCaptureState> {
 public:
  Maybe<void> Capture(MultiplyCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const MultiplyCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Multiply::Capture(MultiplyCaptureState* state, const TensorTuple& inputs,
                              const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  state->requires_grad_x = inputs.at(0)->requires_grad();
  state->requires_grad_y = inputs.at(1)->requires_grad();
  if (state->requires_grad_x) { state->index_y = state->SaveTensorForBackward(inputs.at(1)); }
  if (state->requires_grad_y) { state->index_x = state->SaveTensorForBackward(inputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> Multiply::Apply(const MultiplyCaptureState* state, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(2);
  if (state->requires_grad_x) {
    const auto& y = state->SavedTensors().at(state->index_y);
    in_grads->at(0) = JUST(functional::Mul(out_grads.at(0), y));
  }
  if (state->requires_grad_y) {
    const auto& x = state->SavedTensors().at(state->index_x);
    in_grads->at(1) = JUST(functional::Mul(out_grads.at(0), x));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("multiply", Multiply);

}  // namespace one
}  // namespace oneflow
