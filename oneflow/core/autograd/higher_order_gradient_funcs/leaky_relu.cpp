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

struct LeakyReluGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  float alpha = 0.01;
};

class LeakyReluGradGrad : public OpExprGradFunction<LeakyReluGradGradCaptureState> {
  // leaky_relu_grad = (x > 0 ? 1 : alpha) * grad
  // So: out_grad_grad = (x > 0 ? 1 : alpha) * gradgrad
  //     x_grad_grad = 0 * gradgrad = 0
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(LeakyReluGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<float>("alpha"));
    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LeakyReluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(1) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("leaky_relu_grad", LeakyReluGradGrad);

}  // namespace one
}  // namespace oneflow
