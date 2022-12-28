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

struct BinaryCrossEntropyCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  bool has_weight = false;
};

class BinaryCrossEntropy : public OpExprGradFunction<BinaryCrossEntropyCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BinaryCrossEntropy::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> BinaryCrossEntropy::Capture(BinaryCrossEntropyCaptureState* ctx,
                                        const TensorTuple& inputs, const TensorTuple& outputs,
                                        const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() >= 2 && inputs.size() <= 3);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = inputs[0]->requires_grad();
  ctx->target_requires_grad = inputs[1]->requires_grad();
  ctx->has_weight = inputs.size() == 3;

  ctx->SaveTensorForBackward(inputs[0]);  // input
  ctx->SaveTensorForBackward(inputs[1]);  // target
  if (ctx->has_weight) {
    ctx->SaveTensorForBackward(inputs[2]);  // weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropy::Apply(const BinaryCrossEntropyCaptureState* ctx,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(),
                     2 + ctx->has_weight);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(2 + ctx->has_weight);

  const auto& dy = out_grads[0];
  const auto& input = ctx->SavedTensors()[0];
  const auto& target = ctx->SavedTensors()[1];
  const auto& weight = ctx->has_weight ? Optional<one::Tensor>(ctx->SavedTensors()[2]) : NullOpt;

  if (ctx->input_requires_grad) {
    (*in_grads)[0] = JUST(functional::BinaryCrossEntropyLossGrad(dy, input, target, weight));
  }
  if (ctx->target_requires_grad) {
    (*in_grads)[1] = JUST(functional::BinaryCrossEntropyLossTargetGrad(dy, input, target, weight));
  }
  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy", BinaryCrossEntropy);
}  // namespace one
}  // namespace oneflow
