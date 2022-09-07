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

struct NLLProbCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double label_smoothing = 0;
};

class NLLProbGradFunction : public OpExprGradFunction<NLLProbCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(NLLProbCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const NLLProbCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> NLLProbGradFunction::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> NLLProbGradFunction::Capture(NLLProbCaptureState* ctx, const TensorTuple& inputs,
                                         const TensorTuple& outputs, const AttrMap& attrs) const {
  auto input = JUST(VectorAt(inputs, 0));
  ctx->requires_grad = input->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->label_smoothing = JUST(composed_attrs.GetAttr<double>("label_smoothing"));
  ctx->SaveTensorForBackward(input);                      // input
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // target
  if (inputs.size() == 3) {
    ctx->SaveTensorForBackward(inputs[2]);  // weight
  }
  return Maybe<void>::Ok();
}

Maybe<void> NLLProbGradFunction::Apply(const NLLProbCaptureState* ctx, const TensorTuple& out_grads,
                                       TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  // CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_GE_OR_RETURN(ctx->SavedTensors().size(), 2)
      << Error::RuntimeError()
      << "The number of saved tensors is expected to be greater than or equal to 2, but got "
      << ctx->SavedTensors().size();
  const auto& out_grad = out_grads[0];
  const auto& input = ctx->SavedTensors()[0];
  const auto& target = ctx->SavedTensors()[1];

  in_grads->resize(ctx->SavedTensors().size());

  if (ctx->SavedTensors().size() == 2) {
    JUST(VectorAt(*in_grads, 0)) =
        JUST(functional::NLLProbGrad(out_grad, input, target, NullOpt, ctx->label_smoothing));
  } else {
    // has weight
    auto weight = JUST(VectorAt(ctx->SavedTensors(), 2));
    JUST(VectorAt(*in_grads, 0)) =
        JUST(functional::NLLProbGrad(out_grad, input, target, weight, ctx->label_smoothing));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("nll_prob", NLLProbGradFunction);

}  // namespace one

}  // namespace oneflow
