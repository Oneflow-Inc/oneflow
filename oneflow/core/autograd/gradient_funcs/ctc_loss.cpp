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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct CTCLossCaptureState : public AutoGradCaptureState {
  int64_t max_target_length;
  int32_t blank;
  bool zero_infinity;
  bool requires_grad;
};

class CTCLoss : public OpExprGradFunction<CTCLossCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(CTCLossCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const CTCLossCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> CTCLoss::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> CTCLoss::Capture(CTCLossCaptureState* ctx, const TensorTuple& inputs,
                             const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->max_target_length = JUST(composed_attrs.GetAttr<int64_t>("max_target_length"));
  ctx->blank = JUST(composed_attrs.GetAttr<int64_t>("blank"));
  ctx->zero_infinity = JUST(composed_attrs.GetAttr<bool>("zero_infinity"));

  CHECK_EQ_OR_RETURN(inputs.size(), 4);       // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 2);      // NOLINT(maybe-need-error-msg)
  ctx->SaveTensorForBackward(outputs.at(0));  // loss
  ctx->SaveTensorForBackward(outputs.at(1));  // alpha
  ctx->SaveTensorForBackward(inputs.at(0));   // log_probs
  ctx->SaveTensorForBackward(inputs.at(1));   // targets
  ctx->SaveTensorForBackward(inputs.at(2));   // input_lengths
  ctx->SaveTensorForBackward(inputs.at(3));   // target_lengths
  return Maybe<void>::Ok();
}

Maybe<void> CTCLoss::Apply(const CTCLossCaptureState* ctx, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)

  const auto& grad_out = out_grads.at(0);
  const auto& loss = ctx->SavedTensors().at(0);
  const auto& alpha = ctx->SavedTensors().at(1);
  const auto& log_probs = ctx->SavedTensors().at(2);
  const auto& targets = ctx->SavedTensors().at(3);
  const auto& input_lengths = ctx->SavedTensors().at(4);
  const auto& target_lengths = ctx->SavedTensors().at(5);
  in_grads->resize(4);
  in_grads->at(0) = JUST(functional::CtcLossGrad(grad_out, log_probs, targets, input_lengths,
                                                 target_lengths, loss, alpha, ctx->blank,
                                                 ctx->zero_infinity, ctx->max_target_length));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("ctc_loss", CTCLoss);

}  // namespace one
}  // namespace oneflow