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

struct BinaryCrossEntropyWithLogitsCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  bool has_pos_weight = false;
};

class BinaryCrossEntropyWithLogits
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsCaptureState> {
 public:
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsCaptureState* state,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;
};
Maybe<void> BinaryCrossEntropyWithLogits::Capture(BinaryCrossEntropyWithLogitsCaptureState* state,
                                                  const TensorTuple& inputs,
                                                  const TensorTuple& outputs,
                                                  const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* op_ctx = dynamic_cast<const BinaryCrossEntropyWithLogitsOp*>(ctx);
  state->has_pos_weight = op_ctx->has_pos_weight();
  state->SaveTensorForBackward(inputs.at(0));  // input
  state->SaveTensorForBackward(inputs.at(1));  // target
  if (inputs.size() == 3) {
    state->SaveTensorForBackward(inputs.at(2));  // weight or pos_weight
  }
  if (inputs.size() == 4) {
    state->SaveTensorForBackward(inputs.at(2));  // weight
    state->SaveTensorForBackward(inputs.at(3));  // pos_weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogits::Apply(
    const BinaryCrossEntropyWithLogitsCaptureState* state, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);
  const auto& input = state->SavedTensors().at(0);
  const auto& target = state->SavedTensors().at(1);

  in_grads->resize(state->SavedTensors().size());

  if (state->SavedTensors().size() == 3) {
    if (state->has_pos_weight) {
      const auto& pos_weight = state->SavedTensors().at(2);
      in_grads->at(0) = JUST(
          functional::BinaryCrossEntropyWithLogitsLossGrad(dy, input, target, NullOpt, pos_weight));
    } else {
      const auto& weight = state->SavedTensors().at(2);
      in_grads->at(0) = JUST(
          functional::BinaryCrossEntropyWithLogitsLossGrad(dy, input, target, weight, NullOpt));
    }
  } else if (state->SavedTensors().size() == 4) {
    const auto& weight = state->SavedTensors().at(2);
    const auto& pos_weight = state->SavedTensors().at(3);
    in_grads->at(0) = JUST(
        functional::BinaryCrossEntropyWithLogitsLossGrad(dy, input, target, weight, pos_weight));
  } else {
    in_grads->at(0) =
        JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(dy, input, target, NullOpt, NullOpt));
  }
  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits", BinaryCrossEntropyWithLogits);
}  // namespace one
}  // namespace oneflow
