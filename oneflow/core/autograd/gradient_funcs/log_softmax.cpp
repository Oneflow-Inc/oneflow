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
  Maybe<void> Capture(LogSoftmaxCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const LogSoftmaxCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> LogSoftmax::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  grad_op_ = JUST(one::OpBuilder("log_softmax_grad", GradientOpName(op_name))
                      .Input("prob")
                      .Input("dy")
                      .Output("dx")
                      .Build());
  return Maybe<void>::Ok();
}

Maybe<void> LogSoftmax::Capture(LogSoftmaxCaptureState* state, const TensorTuple& inputs,
                                const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  state->requires_grad = inputs.at(0)->requires_grad();

  if (!state->requires_grad) return Maybe<void>::Ok();

  state->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> LogSoftmax::Apply(const LogSoftmaxCaptureState* state, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  if (!state->requires_grad) return Maybe<void>::Ok();
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);
  const auto& prob = state->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {prob, dy}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("log_softmax", LogSoftmax);

}  // namespace one
}  // namespace oneflow
