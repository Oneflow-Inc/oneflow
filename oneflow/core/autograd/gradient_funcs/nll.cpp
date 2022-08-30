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
  bool requires_grad = false;
  int64_t ignore_index = -100;
};

class NLLGradFunction : public OpExprGradFunction<NLLCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(NLLCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const NLLCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> NLLGradFunction::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> NLLGradFunction::Capture(NLLCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  auto input = JUST(VectorAt(inputs, 0));
  ctx->requires_grad = input->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->ignore_index = JUST(composed_attrs.GetAttr<int64_t>("ignore_index"));
  ctx->SaveTensorForBackward(input);                      // input
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // target
  if (inputs.size() == 3) {
    ctx->SaveTensorForBackward(inputs[2]);  // weight
  }
  return Maybe<void>::Ok();
}

Maybe<void> NLLGradFunction::Apply(const NLLCaptureState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
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
        JUST(functional::NLLGrad(out_grad, input, target, NullOpt, ctx->ignore_index));
  } else {
    // has weight
    auto weight = JUST(VectorAt(ctx->SavedTensors(), 2));
    JUST(VectorAt(*in_grads, 0)) =
        JUST(functional::NLLGrad(out_grad, input, target, weight, ctx->ignore_index));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("nll", NLLGradFunction);

}  // namespace one

}  // namespace oneflow
