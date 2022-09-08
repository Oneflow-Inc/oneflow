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

namespace oneflow {

namespace one {

struct NLLCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool grad_requires_grad = false;
  bool has_weight = false;
  int64_t ignore_index = -100;
};

class NLLLossGradGrad : public OpExprGradFunction<NLLCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(NLLCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const NLLCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> NLLLossGradGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> NLLLossGradGrad::Capture(NLLCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  // dy, input, target[, weight]
  CHECK_OR_RETURN(inputs.size() >= 3 && inputs.size() <= 4);  // NOLINT(maybe-need-error-msg)
  ctx->grad_requires_grad = inputs[0]->requires_grad();
  ctx->input_requires_grad = inputs[1]->requires_grad();
  ctx->has_weight = inputs.size() == 4;

  if (ctx->grad_requires_grad) {
    ctx->SaveTensorForBackward(inputs[2]);
    if (ctx->has_weight) { ctx->SaveTensorForBackward(inputs[3]); }  // weight
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->ignore_index = JUST(composed_attrs.GetAttr<int64_t>("ignore_index"));
  }

  return Maybe<void>::Ok();
}

Maybe<void> NLLLossGradGrad::Apply(const NLLCaptureState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3 + ctx->has_weight);

  if (ctx->grad_requires_grad) {
    const auto& target = JUST(VectorAt(ctx->SavedTensors(), 0));
    if (ctx->has_weight) {
      auto weight = JUST(VectorAt(ctx->SavedTensors(), 1));
      (*in_grads)[0] =
          JUST(functional::NLLLoss(out_grads[0], target, weight, ctx->ignore_index, "none"));
    } else {
      (*in_grads)[0] =
          JUST(functional::NLLLoss(out_grads[0], target, NullOpt, ctx->ignore_index, "none"));
    }
  }
  if (ctx->input_requires_grad) { (*in_grads)[1] = JUST(functional::ZerosLike(out_grads[0])); }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("nll_grad", NLLLossGradGrad);

}  // namespace one

}  // namespace oneflow
