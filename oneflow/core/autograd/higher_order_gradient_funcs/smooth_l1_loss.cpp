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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct SmoothL1LossGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  size_t grad_index = 0;
  size_t input_index = 0;
  size_t target_index = 0;
  float beta = 0.0;
};

class SmoothL1LossGradGrad : public OpExprGradFunction<SmoothL1LossGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SmoothL1LossGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // grad, input, target
    CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)

    ctx->grad_requires_grad = inputs[0]->requires_grad();
    ctx->input_requires_grad = inputs[1]->requires_grad();
    ctx->target_requires_grad = inputs[2]->requires_grad();

    if (ctx->input_requires_grad || ctx->target_requires_grad) {
      ctx->grad_index = ctx->SaveTensorForBackward(inputs[0]);
    }
    ctx->input_index = ctx->SaveTensorForBackward(inputs[1]);
    ctx->target_index = ctx->SaveTensorForBackward(inputs[2]);

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->beta = JUST(composed_attrs.GetAttr<float>("beta"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SmoothL1LossGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(3);
    const auto& input = JUST(VectorAt(ctx->SavedTensors(), ctx->input_index));
    const auto& target = JUST(VectorAt(ctx->SavedTensors(), ctx->target_index));

    if (ctx->grad_requires_grad) {
      (*in_grads)[0] = JUST(functional::SmoothL1LossGrad(out_grads[0], input, target, ctx->beta));
    }
    if (ctx->input_requires_grad || ctx->target_requires_grad) {
      const auto& grad = JUST(VectorAt(ctx->SavedTensors(), ctx->grad_index));
      auto condition = JUST(functional::sequence_function(functional::Sub)
                                .then(functional::Abs)
                                .then([&ctx](const std::shared_ptr<Tensor>& input) {
                                  return functional::ScalarLogicalLess(input, ctx->beta);
                                })
                                .call(input, target, /*alpha=*/1, /*inplace=*/false));
      auto out = JUST(functional::sequence_function(functional::Mul)
                          .then(std::bind(functional::Mul, std::placeholders::_1, condition))
                          .then([&ctx](const std::shared_ptr<Tensor>& input) {
                            double inv_beta = ctx->beta == 0.0 ? 0.0 : 1.0 / ctx->beta;
                            return functional::ScalarMul(inv_beta, input);
                          })
                          .call(out_grads[0], grad));
      if (ctx->input_requires_grad) { (*in_grads)[1] = out; }
      if (ctx->target_requires_grad) { (*in_grads)[2] = JUST(functional::Negative(out)); }
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("smooth_l1_loss_grad", SmoothL1LossGradGrad);

}  // namespace one
}  // namespace oneflow
