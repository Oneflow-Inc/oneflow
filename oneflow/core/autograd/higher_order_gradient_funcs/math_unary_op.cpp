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
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct UnaryGradGradState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
};

class SinGradGrad : public OpExprGradFunction<UnaryGradGradState> {
  // sin_grad = cos(x) * grad
  // So: out_grad_grad = cos(x) * gradgrad
  //     x_grad_grad = -sin(x) * grad * gradgrad
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryGradGradState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2) << "SinGradGrad op have 2 inputs";
    CHECK_EQ_OR_RETURN(outputs.size(), 1) << "SinGradGrad op have 1 output";
    ctx->x_requires_grad = inputs[0]->requires_grad();
    ctx->grad_requires_grad = inputs[1]->requires_grad();
    ctx->SaveTensorForBackward(inputs[0]);
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs[1]); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryGradGradState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors()[0];
    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors()[1];
      (*in_grads)[0] =
          JUST(functional::sequence_function(functional::SinGradGrad)
                   .then(std::bind(functional::Mul, out_grads[0], std::placeholders::_1))
                   .call(x, grad));
    }
    if (ctx->grad_requires_grad) { (*in_grads)[1] = JUST(functional::SinGrad(x, out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

class CosGradGrad : public OpExprGradFunction<UnaryGradGradState> {
  // sin_grad = -sin(x) * grad
  // So: out_grad_grad = -sin(x) * gradgrad
  //     x_grad_grad = -cos(x) * grad * gradgrad
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryGradGradState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2) << "CosGradGrad op have 2 inputs";
    CHECK_EQ_OR_RETURN(outputs.size(), 1) << "CosGradGrad op have 1 output";
    ctx->x_requires_grad = inputs[0]->requires_grad();
    ctx->grad_requires_grad = inputs[1]->requires_grad();
    ctx->SaveTensorForBackward(inputs[0]);
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs[1]); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryGradGradState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors()[0];
    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors()[1];
      (*in_grads)[0] =
          JUST(functional::sequence_function(functional::CosGradGrad)
                   .then(std::bind(functional::Mul, out_grads[0], std::placeholders::_1))
                   .call(x, grad));
    }
    if (ctx->grad_requires_grad) { (*in_grads)[1] = JUST(functional::CosGrad(x, out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

class NegativeGradGrad : public OpExprGradFunction<UnaryGradGradState> {
  // neg_grad = -1 * grad
  // So: out_grad_grad = -1 * gradgrad
  //     x_grad_grad = 0 * gradgrad = 0
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryGradGradState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryGradGradState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) { in_grads->at(1) = JUST(functional::Negative(out_grads.at(0))); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("sin_grad", SinGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("cos_grad", CosGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("negative_grad", NegativeGradGrad);

}  // namespace one
}  // namespace oneflow
