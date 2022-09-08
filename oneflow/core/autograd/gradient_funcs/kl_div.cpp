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

namespace oneflow {
namespace one {

struct KLDivLossCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  bool log_target = false;
};

class KLDivLoss : public OpExprGradFunction<KLDivLossCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(KLDivLossCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const KLDivLossCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> KLDivLoss::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
Maybe<void> KLDivLoss::Capture(KLDivLossCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = inputs[0]->requires_grad();
  ctx->target_requires_grad = inputs[1]->requires_grad();

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->log_target = JUST(composed_attrs.GetAttr<bool>("log_target"));
  ctx->SaveTensorForBackward(inputs[0]);  // input
  ctx->SaveTensorForBackward(inputs[1]);  // target
  return Maybe<void>::Ok();
}
Maybe<void> KLDivLoss::Apply(const KLDivLossCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);            // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(), 2);  // NOLINT(maybe-need-error-msg)
  const auto& dy = out_grads[0];
  const auto& input = ctx->SavedTensors()[0];
  const auto& target = ctx->SavedTensors()[1];
  in_grads->resize(2);

  if (ctx->input_requires_grad) {
    (*in_grads)[0] = JUST(functional::KLDivLossGrad(dy, input, target, ctx->log_target));
  }
  if (ctx->target_requires_grad) {
    (*in_grads)[1] = JUST(functional::KLDivLossTargetGrad(dy, input, target, ctx->log_target));
  }

  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("kl_div_loss", KLDivLoss);

}  // namespace one
}  // namespace oneflow
