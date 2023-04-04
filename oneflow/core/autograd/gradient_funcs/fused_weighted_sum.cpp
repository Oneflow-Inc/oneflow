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

namespace oneflow {
namespace one {

struct FusedWeightedSumCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
  std::vector<float> weights;
  float alpha{};
};

class FusedWeightedSum : public OpExprGradFunction<FusedWeightedSumCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FusedWeightedSumCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->requires_grad.resize(inputs.size());
    ctx->weights = JUST(attrs.GetAttr<std::vector<float>>("weights"));
    ctx->alpha = JUST(attrs.GetAttr<float>("alpha"));
    CHECK_EQ_OR_RETURN(ctx->weights.size(), inputs.size());
    for (int i = 0; i < inputs.size(); ++i) { ctx->requires_grad[i] = inputs[i]->requires_grad(); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedWeightedSumCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(ctx->requires_grad.size());
    for (int i = 0; i < ctx->requires_grad.size(); ++i) {
      if (ctx->requires_grad[i]) {
        (*in_grads)[i] =
            JUST(functional::ScalarMul(out_grads[0], ctx->weights[i] * ctx->alpha, false));
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_weighted_sum", FusedWeightedSum);

}  // namespace one
}  // namespace oneflow
