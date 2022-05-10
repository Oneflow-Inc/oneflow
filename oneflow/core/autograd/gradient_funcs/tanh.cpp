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
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct TanhCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
};

class TanhGrad : public OpExprGradFunction<TanhCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(TanhCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const TanhCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    const auto& y = ctx->SavedTensors().at(0);
    const auto& a = functional::Mul(y, y);
    const auto& aa = functional::ScalarSub(1, JUST(a));
    in_grads->at(0) = JUST(functional::Mul(out_grads.at(0), JUST(aa)));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tanh", TanhGrad);

}  // namespace one
}  // namespace oneflow
