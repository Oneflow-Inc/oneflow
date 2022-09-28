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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

struct LogSoftmaxCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class LogSoftmax : public OpExprGradFunction<LogSoftmaxCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(LogSoftmaxCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const LogSoftmaxCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> LogSoftmax::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> LogSoftmax::Capture(LogSoftmaxCaptureState* ctx, const TensorTuple& inputs,
                                const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  ctx->requires_grad = inputs.at(0)->requires_grad();
  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> LogSoftmax::Apply(const LogSoftmaxCaptureState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& dy = out_grads.at(0);
  const auto& y = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::LogSoftmaxGrad(dy, y));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("log_softmax", LogSoftmax);

}  // namespace one
}  // namespace oneflow
