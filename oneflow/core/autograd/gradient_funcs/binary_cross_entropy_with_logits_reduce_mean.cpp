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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BinaryCrossEntropyWithLogitsReduceMeanCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool target_requires_grad = false;
};

class BinaryCrossEntropyWithLogitsReduceMean
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsReduceMeanCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx,
                      const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;
};

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Capture(
    BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx, const TensorTuple& inputs,
    const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
  ctx->target_requires_grad = JUST(VectorAt(inputs, 1))->requires_grad();

  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // target
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Apply(
    const BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& dy = JUST(VectorAt(out_grads, 0));
  const auto& input = JUST(VectorAt(ctx->SavedTensors(), 0));
  const auto& target = JUST(VectorAt(ctx->SavedTensors(), 1));
  in_grads->resize(2);

  if (ctx->input_requires_grad) {
    (*in_grads)[0] =
        JUST(functional::BinaryCrossEntropyWithLogitsReduceMeanLossGrad(dy, input, target));
  }
  if (ctx->target_requires_grad) {
    (*in_grads)[1] =
        JUST(functional::BinaryCrossEntropyWithLogitsReduceMeanLossTargetGrad(dy, input, target));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits_reduce_mean",
                               BinaryCrossEntropyWithLogitsReduceMean);

}  // namespace one

}  // namespace oneflow
