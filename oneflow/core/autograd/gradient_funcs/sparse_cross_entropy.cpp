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

struct SparseCrossEntropyCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  int64_t depth = -1;
  size_t prediction_index = -1;
  size_t label_index = -1;
};

template<bool is_distributed>
class SparseCrossEntropy : public OpExprGradFunction<SparseCrossEntropyCaptureState> {
 public:
  Maybe<void> Capture(SparseCrossEntropyCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<SparseCrossEntropyOp>());
    state->depth = op_ctx->depth();
    state->prediction_index = state->SaveTensorForBackward(inputs.at(0));  // prediction
    state->label_index = state->SaveTensorForBackward(inputs.at(1));       // label
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SparseCrossEntropyCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& prediction = state->SavedTensors().at(state->prediction_index);
    const auto& label = state->SavedTensors().at(state->label_index);
    in_grads->resize(2);
    if (is_distributed) {
      in_grads->at(0) = JUST(
          functional::SparseCrossEntropyMsGrad(prediction, label, out_grads.at(0), state->depth));
    } else {
      in_grads->at(0) = JUST(
          functional::SparseCrossEntropyGrad(prediction, label, out_grads.at(0), state->depth));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_cross_entropy_ms", SparseCrossEntropy<true>);
REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_cross_entropy", SparseCrossEntropy<false>);

}  // namespace one
}  // namespace oneflow
