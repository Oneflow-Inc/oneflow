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

struct CombinedMarginLossCaptureState : public AutoGradCaptureState {
  float m1;
  float m2;
  float m3;
  int64_t depth;
  size_t label_index;
  size_t theta_index;
  bool requires_grad;
};

class CombinedMarginLoss : public OpExprGradFunction<CombinedMarginLossCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CombinedMarginLossCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);                // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();  // x
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->label_index = ctx->SaveTensorForBackward(inputs.at(1));   // label
    ctx->theta_index = ctx->SaveTensorForBackward(outputs.at(1));  // theta

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->m1 = JUST(composed_attrs.GetAttr<float>("m1"));
    ctx->m2 = JUST(composed_attrs.GetAttr<float>("m2"));
    ctx->m3 = JUST(composed_attrs.GetAttr<float>("m3"));
    ctx->depth = JUST(composed_attrs.GetAttr<int64_t>("depth"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CombinedMarginLossCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);

    if (ctx->requires_grad) {
      const auto& label = ctx->SavedTensors().at(ctx->label_index);
      const auto& theta = ctx->SavedTensors().at(ctx->theta_index);
      in_grads->at(0) = JUST(functional::CombinedMarginLossGrad(
          out_grads.at(0), label, theta, ctx->m1, ctx->m2, ctx->m3, ctx->depth));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("combined_margin_loss", CombinedMarginLoss);

}  // namespace one
}  // namespace oneflow
