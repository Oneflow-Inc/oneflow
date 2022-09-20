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
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct BinaryCrossEntropyGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  bool has_weight = false;
};

class BinaryCrossEntropyGradGrad
    : public OpExprGradFunction<BinaryCrossEntropyGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BinaryCrossEntropyGradGrad::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> BinaryCrossEntropyGradGrad::Capture(BinaryCrossEntropyGradGradCaptureState* ctx,
                                                const TensorTuple& inputs,
                                                const TensorTuple& outputs,
                                                const AttrMap& attrs) const {
  // dy, input, target[, weight]
  CHECK_OR_RETURN(inputs.size() >= 3 && inputs.size() <= 4);  // NOLINT(maybe-need-error-msg)
  ctx->grad_requires_grad = inputs[0]->requires_grad();
  ctx->input_requires_grad = inputs[1]->requires_grad();
  ctx->target_requires_grad = inputs[2]->requires_grad();
  ctx->has_weight = inputs.size() == 4;

  ctx->SaveTensorForBackward(inputs[0]);  // grad
  ctx->SaveTensorForBackward(inputs[1]);  // input
  ctx->SaveTensorForBackward(inputs[2]);  // target
  if (ctx->has_weight) {
    ctx->SaveTensorForBackward(inputs[3]);  // weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyGradGrad::Apply(const BinaryCrossEntropyGradGradCaptureState* ctx,
                                              const TensorTuple& out_grads,
                                              TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(),
                     3 + ctx->has_weight);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3 + ctx->has_weight);
  const auto& grad = ctx->SavedTensors()[0];
  const auto& input = ctx->SavedTensors()[1];
  const auto& target = ctx->SavedTensors()[2];

  // dx = grad * [-target/input + (1-target)/(1-input)]
  // grad_for_grad = out_grad * [-target/input + (1-target)/(1-input)]
  // grad_for_input = out_grad * grad * [target/(input*input) + (1-target)/((1-input)*(1-input))]
  //                = out_grad * grad * [(input*input-2*input*target+target)/(input*(1-input))^2]
  // grad_for_target = out_grad * grad * [1/(input*(1-input))]
  if (ctx->grad_requires_grad) {
    const auto& weight = ctx->has_weight ? Optional<one::Tensor>(ctx->SavedTensors()[3]) : NullOpt;
    (*in_grads)[0] =
        JUST(functional::BinaryCrossEntropyLossGrad(out_grads[0], input, target, weight));
  }
  if (ctx->input_requires_grad) {
    auto one_sub_input = JUST(functional::ScalarSub(1, input, /*alpha=*/1));
    auto input_mul_target = JUST(functional::Mul(input, target));
    auto numerator =
        JUST(functional::sequence_function(functional::Square)
                 .then(std::bind(functional::Sub, std::placeholders::_1, input_mul_target,
                                 /*alpha=*/2, /*inplace=*/false))
                 .then([&target](const std::shared_ptr<Tensor>& in) {
                   return functional::Add(in, target, /*alpha=*/1, /*inplace=*/false);
                 })
                 .call(input));
    auto res = JUST(functional::sequence_function(functional::Mul)
                        .then(functional::Square)
                        .then(std::bind(functional::Div, numerator, std::placeholders::_1))
                        .then(std::bind(functional::Mul, std::placeholders::_1, out_grads[0]))
                        .then(std::bind(functional::Mul, std::placeholders::_1, grad))
                        .call(input, one_sub_input));
    (*in_grads)[1] = ctx->has_weight ? JUST(functional::Mul(ctx->SavedTensors()[3], res)) : res;
  }
  if (ctx->target_requires_grad) {
    auto input_sub_one = JUST(functional::ScalarAdd(-1, input, /*alpha=*/1));
    auto res = JUST(functional::sequence_function(functional::Mul)
                        .then(std::bind(functional::LogGrad, std::placeholders::_1, out_grads[0]))
                        .then(std::bind(functional::Mul, std::placeholders::_1, grad))
                        .call(input, input_sub_one));
    (*in_grads)[2] = ctx->has_weight ? JUST(functional::Mul(ctx->SavedTensors()[3], res)) : res;
  }

  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_grad", BinaryCrossEntropyGradGrad);
}  // namespace one
}  // namespace oneflow
