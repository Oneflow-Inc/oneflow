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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SmoothL1LossCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  float beta = 0.0;
};

class SmoothL1Loss : public OpExprGradFunction<SmoothL1LossCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SmoothL1LossCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)

    ctx->input_requires_grad = inputs.at(0)->requires_grad();   // input
    ctx->target_requires_grad = inputs.at(1)->requires_grad();  // target

    ctx->SaveTensorForBackward(inputs.at(0));  // input
    ctx->SaveTensorForBackward(inputs.at(1));  // target

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->beta = JUST(composed_attrs.GetAttr<float>("beta"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SmoothL1LossCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);            // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(), 2);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);
    const auto& input = ctx->SavedTensors().at(0);
    const auto& target = ctx->SavedTensors().at(1);
    const auto& grad = JUST(functional::SmoothL1LossGrad(out_grads[0], input, target, ctx->beta));

    if (ctx->input_requires_grad) { (*in_grads)[0] = grad; }
    if (ctx->target_requires_grad) { (*in_grads)[1] = JUST(functional::Negative(grad)); }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("smooth_l1_loss", SmoothL1Loss);  // todo: name

}  // namespace one
}  // namespace oneflow
