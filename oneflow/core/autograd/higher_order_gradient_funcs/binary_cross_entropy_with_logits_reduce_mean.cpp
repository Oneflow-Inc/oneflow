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
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool target_requires_grad = false;

  size_t grad_index = 0;
  size_t input_index = 0;
  size_t target_index = 0;
};

class BinaryCrossEntropyWithLogitsReduceMeanGradGrad
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState* ctx,
                      const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;
};

Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradGrad::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradGrad::Capture(
    BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState* ctx, const TensorTuple& inputs,
    const TensorTuple& outputs, const AttrMap& attrs) const {
  // dy, input, target
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)
  ctx->grad_requires_grad = inputs[0]->requires_grad();
  ctx->input_requires_grad = inputs[1]->requires_grad();
  ctx->target_requires_grad = inputs[2]->requires_grad();

  if (ctx->input_requires_grad || ctx->target_requires_grad) {
    ctx->grad_index = ctx->SaveTensorForBackward(inputs[0]);  // grad
  }
  if (ctx->input_requires_grad || ctx->grad_requires_grad) {
    ctx->input_index = ctx->SaveTensorForBackward(inputs[1]);  // input
  }
  if (ctx->grad_requires_grad) {
    ctx->target_index = ctx->SaveTensorForBackward(inputs[2]);  // target
  }
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMeanGradGrad::Apply(
    const BinaryCrossEntropyWithLogitsReduceMeanGradGradCaptureState* ctx,
    const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3);

  // dx = grad * weight * (input.sigmoid() - target)
  // grad_for_input = out_grad * grad * weight * sig * (1-sig)
  // grad_for_target = -out_grad * grad * weight
  if (ctx->grad_requires_grad) {
    const auto& input = JUST(VectorAt(ctx->SavedTensors(), ctx->input_index));
    const auto& target = JUST(VectorAt(ctx->SavedTensors(), ctx->target_index));
    (*in_grads)[0] = JUST(
        functional::sequence_function(functional::Sigmoid)
            .then(std::bind(functional::Sub, std::placeholders::_1, target, /*alpha=*/1,
                            /*inplace=*/false))
            .then(std::bind(functional::Mul, std::placeholders::_1, out_grads[0]))
            .then(std::bind(functional::ReduceMean, std::placeholders::_1, std::vector<int32_t>{},
                            /*keepdim=*/false))
            .call(input));
  }
  if (ctx->input_requires_grad) {
    const auto& grad = JUST(VectorAt(ctx->SavedTensors(), ctx->grad_index));
    const auto& input = JUST(VectorAt(ctx->SavedTensors(), ctx->input_index));
    const auto& mean_grad = JUST(functional::ScalarMul(1.0 / out_grads[0]->nelement(), grad));
    (*in_grads)[1] =
        JUST(functional::sequence_function(functional::Sigmoid)
                 .then(std::bind(functional::SigmoidGrad, std::placeholders::_1, out_grads[0]))
                 .then(std::bind(functional::Mul, std::placeholders::_1, mean_grad))
                 .call(input));
  }
  if (ctx->target_requires_grad) {
    const auto& grad = JUST(VectorAt(ctx->SavedTensors(), ctx->grad_index));
    const auto& mean_grad = JUST(functional::ScalarMul(1.0 / out_grads[0]->nelement(), grad));
    (*in_grads)[2] = JUST(functional::sequence_function(functional::Mul)
                              .then(functional::Negative)
                              .call(out_grads[0], mean_grad));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits_reduce_mean_grad",
                               BinaryCrossEntropyWithLogitsReduceMeanGradGrad);

}  // namespace one

}  // namespace oneflow
