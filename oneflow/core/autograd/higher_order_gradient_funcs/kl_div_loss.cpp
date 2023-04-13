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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct KLDivLossGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool target_requires_grad = false;
  bool log_target = false;

  size_t input_index = 0;
  size_t target_index = 0;
};

class KLDivLossGradGrad : public OpExprGradFunction<KLDivLossGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(KLDivLossGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const KLDivLossGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> KLDivLossGradGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
Maybe<void> KLDivLossGradGrad::Capture(KLDivLossGradGradCaptureState* ctx,
                                       const TensorTuple& inputs, const TensorTuple& outputs,
                                       const AttrMap& attrs) const {
  // grad, input, target
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)
  ctx->grad_requires_grad = inputs[0]->requires_grad();
  ctx->input_requires_grad = inputs[1]->requires_grad();
  ctx->target_requires_grad = inputs[2]->requires_grad();

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->log_target = JUST(composed_attrs.GetAttr<bool>("log_target"));

  ctx->input_index = ctx->SaveTensorForBackward(inputs[1]);   // input
  ctx->target_index = ctx->SaveTensorForBackward(inputs[2]);  // target

  return Maybe<void>::Ok();
}
Maybe<void> KLDivLossGradGrad::Apply(const KLDivLossGradGradCaptureState* ctx,
                                     const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3);

  if (ctx->grad_requires_grad) {
    const auto& input = JUST(VectorAt(ctx->SavedTensors(), ctx->input_index));
    const auto& target = JUST(VectorAt(ctx->SavedTensors(), ctx->target_index));
    (*in_grads)[0] = JUST(functional::KLDivLossGrad(out_grads[0], input, target, ctx->log_target));
  }
  if (ctx->input_requires_grad) { (*in_grads)[1] = JUST(functional::ZerosLike(out_grads[0])); }
  if (ctx->target_requires_grad) { (*in_grads)[2] = JUST(functional::ZerosLike(out_grads[0])); }
  //// In pytorch 1.13 the higher derivative grad is fixed, which will cause difference here
  // if (ctx->target_requires_grad) {
  //   if (ctx->log_target) (*in_grads)[2] =
  //   JUST(functional::Mul(JUST(functional::Negative(JUST(functional::Exp(target)))),
  //   out_grads[0])); else (*in_grads)[2] = JUST(functional::Negative(out_grads[0]));
  // }

  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("kl_div_loss_grad", KLDivLossGradGrad);

}  // namespace one
}  // namespace oneflow
