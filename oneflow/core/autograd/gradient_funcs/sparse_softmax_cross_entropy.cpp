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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SparseSoftmaxCrossEntropyCaptureState : public AutoGradCaptureState {
  int64_t depth;
};

class SparseSoftmaxCrossEntropy : public OpExprGradFunction<SparseSoftmaxCrossEntropyCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SparseSoftmaxCrossEntropyCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const SparseSoftmaxCrossEntropyCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> SparseSoftmaxCrossEntropy::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropy::Capture(SparseSoftmaxCrossEntropyCaptureState* state,
                                               const TensorTuple& inputs,
                                               const TensorTuple& outputs,
                                               const OpInterpCtx* ctx) const {
  auto* interp_ctx = dynamic_cast<const SparseSoftmaxCrossEntropyOpInterpCtx*>(ctx);
  state->depth = interp_ctx->depth;
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  CHECK_EQ_OR_RETURN(outputs.size(), 2);
  state->SaveTensorForBackward(outputs.at(0));  // prob
  state->SaveTensorForBackward(inputs.at(1));   // label
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropy::Apply(const SparseSoftmaxCrossEntropyCaptureState* state,
                                             const TensorTuple& out_grads,
                                             TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);
  const auto& dy = out_grads.at(1);
  const auto& prob = state->SavedTensors().at(0);
  const auto& label = state->SavedTensors().at(1);
  // MutableAttrMap attrs;
  // JUST(attrs.SetAttr<int64_t>("depth", state->depth));
  // SparseSoftmaxCrossEntropy has 2 inputs (prediction and label), and the second input does not
  // require gradient.
  in_grads->resize(2);
  in_grads->at(0) = JUST(functional::SparseSoftmaxCrossEntropyGrad(dy, prob, label, state->depth));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_softmax_cross_entropy", SparseSoftmaxCrossEntropy);

}  // namespace one
}  // namespace oneflow