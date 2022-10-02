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

namespace oneflow {
namespace one {

struct GumbelSoftmaxCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class GumbelSoftmax : public OpExprGradFunction<GumbelSoftmaxCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GumbelSoftmaxCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GumbelSoftmaxCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> GumbelSoftmax::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> GumbelSoftmax::Capture(GumbelSoftmaxCaptureState* ctx, const TensorTuple& inputs,
                             const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  ctx->requires_grad = inputs.at(0)->requires_grad();

  if (!ctx->requires_grad) return Maybe<void>::Ok();

  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> GumbelSoftmax::Apply(const GumbelSoftmaxCaptureState* ctx, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!ctx->requires_grad) return Maybe<void>::Ok();
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& dy = out_grads.at(0);
  const auto& y = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::SoftmaxGrad(dy, y));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("gumbel_softmax", GumbelSoftmax);

}  // namespace one
}  // namespace oneflow
