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

struct BinaryCrossEntropyWithLogitsGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  bool has_weight = false;
  bool has_pos_weight = false;
};

class BinaryCrossEntropyWithLogitsGradGrad
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsGradGradCaptureState* ctx,
                      const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsGradGradCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> BinaryCrossEntropyWithLogitsGradGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogitsGradGrad::Capture(
    BinaryCrossEntropyWithLogitsGradGradCaptureState* ctx, const TensorTuple& inputs,
    const TensorTuple& outputs, const AttrMap& attrs) const {
  // dy, input, target[, weight][, pos_weight]
  CHECK_OR_RETURN(inputs.size() >= 3 && inputs.size() <= 5);  // NOLINT(maybe-need-error-msg)
  ctx->grad_requires_grad = inputs[0]->requires_grad();
  ctx->input_requires_grad = inputs[1]->requires_grad();
  ctx->target_requires_grad = inputs[2]->requires_grad();

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->has_pos_weight = JUST(composed_attrs.GetAttr<bool>("has_pos_weight"));
  ctx->has_weight = inputs.size() == 5 || (inputs.size() == 4 && !ctx->has_pos_weight);
  ctx->SaveTensorForBackward(inputs[0]);  // grad
  ctx->SaveTensorForBackward(inputs[1]);  // input
  ctx->SaveTensorForBackward(inputs[2]);  // target

  if (inputs.size() == 4) {
    ctx->SaveTensorForBackward(inputs[3]);  // weight or pos_weight
  }
  if (inputs.size() == 5) {
    ctx->SaveTensorForBackward(inputs[3]);  // weight
    ctx->SaveTensorForBackward(inputs[4]);  // pos_weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogitsGradGrad::Apply(
    const BinaryCrossEntropyWithLogitsGradGradCaptureState* ctx, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(),
                     3 + ctx->has_weight + ctx->has_pos_weight);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3 + ctx->has_weight + ctx->has_pos_weight);
  const auto& grad = ctx->SavedTensors()[0];
  const auto& input = ctx->SavedTensors()[1];
  const auto& target = ctx->SavedTensors()[2];
  const size_t pos_weight_index = ctx->has_weight ? 4 : 3;
  const auto& weight = ctx->has_weight ? Optional<one::Tensor>(ctx->SavedTensors()[3]) : NullOpt;
  const auto& pos_weight =
      ctx->has_pos_weight ? Optional<one::Tensor>(ctx->SavedTensors()[pos_weight_index]) : NullOpt;

  // dx = grad * weight * (-target*(1-input.sigmoid())*pos_weight + input.sigmoid()*(1-target))
  // grad_for_input = out_grad * grad * weight * sig * (1-sig) * [pos_weight * target + 1 - target]
  // grad_for_target = -out_grad * grad * weight * [pos_weight + sig - pos_weight * sig]
  if (ctx->grad_requires_grad) {
    (*in_grads)[0] = JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(
        out_grads[0], input, target, weight, pos_weight));
  }
  if (ctx->input_requires_grad) {
    auto res = JUST(functional::sequence_function(functional::Sigmoid)
                        .then(std::bind(functional::SigmoidGrad, std::placeholders::_1, grad))
                        .then(std::bind(functional::Mul, std::placeholders::_1, out_grads[0]))
                        .call(input));
    if (ctx->has_pos_weight) {
      res = JUST(functional::sequence_function(functional::Mul)
                     .then([](const std::shared_ptr<Tensor>& input) {
                       return functional::ScalarAdd(1, input, /*alpha=*/Scalar(1));
                     })
                     .then(std::bind(functional::Sub, std::placeholders::_1, target, /*alpha=*/1,
                                     /*inplace=*/false))
                     .then(std::bind(functional::Mul, std::placeholders::_1, res))
                     .call(JUST(pos_weight), target));
    }
    if (ctx->has_weight) { res = JUST(functional::Mul(res, JUST(weight))); }
    (*in_grads)[1] = res;
  }
  if (ctx->target_requires_grad) {
    auto res = JUST(functional::sequence_function(functional::Mul)
                        .then(functional::Negative)
                        .call(out_grads[0], grad));
    if (ctx->has_pos_weight) {
      auto sig = JUST(functional::Sigmoid(input));
      auto one_sub_sig = JUST(functional::ScalarSub(1, sig, /*alpha=*/1));
      res = JUST(functional::sequence_function(functional::Mul)
                     .then([&sig](const std::shared_ptr<Tensor>& input) {
                       return functional::Add(input, sig, /*alpha=*/Scalar(1), /*inplace=*/false);
                     })
                     .then(std::bind(functional::Mul, std::placeholders::_1, res))
                     .call(one_sub_sig, JUST(pos_weight)));
    }
    if (ctx->has_weight) { res = JUST(functional::Mul(res, JUST(weight))); }
    (*in_grads)[2] = res;
  }

  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits_grad",
                               BinaryCrossEntropyWithLogitsGradGrad);
}  // namespace one
}  // namespace oneflow
