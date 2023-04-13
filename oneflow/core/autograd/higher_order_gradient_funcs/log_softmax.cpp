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
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct LogSoftmaxGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool dy_requires_grad = false;
};

class LogSoftmaxGradGrad : public OpExprGradFunction<LogSoftmaxGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(LogSoftmaxGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const LogSoftmaxGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> LogSoftmaxGradGrad::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> LogSoftmaxGradGrad::Capture(LogSoftmaxGradGradCaptureState* ctx,
                                        const TensorTuple& inputs, const TensorTuple& outputs,
                                        const AttrMap& attrs) const {
  // y, dy
  CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  ctx->y_requires_grad = inputs[0]->requires_grad();
  ctx->dy_requires_grad = inputs[1]->requires_grad();

  ctx->SaveTensorForBackward(inputs[0]);
  if (ctx->y_requires_grad) ctx->SaveTensorForBackward(inputs[1]);
  return Maybe<void>::Ok();
}

Maybe<void> LogSoftmaxGradGrad::Apply(const LogSoftmaxGradGradCaptureState* ctx,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  const auto& y = ctx->SavedTensors()[0];
  const std::vector<int32_t> reduce_axis{static_cast<int32_t>(y->ndim() - 1)};

  if (ctx->y_requires_grad) {
    const auto& dy = ctx->SavedTensors()[1];
    in_grads->at(0) =
        JUST(functional::sequence_function(functional::ReduceSum)
                 .then(std::bind(functional::Mul, std::placeholders::_1, out_grads[0]))
                 .then(std::bind(functional::Mul, std::placeholders::_1, JUST(functional::Exp(y))))
                 .then(functional::Negative)
                 .call(dy, reduce_axis, true));
  }
  if (ctx->dy_requires_grad) {
    in_grads->at(1) =
        JUST(functional::sequence_function(functional::Exp)
                 .then(std::bind(functional::Mul, std::placeholders::_1, out_grads[0]))
                 .then(std::bind(functional::ReduceSum, std::placeholders::_1, reduce_axis,
                                 /*keepdim=*/true))
                 .then(std::bind(functional::Sub, out_grads[0], std::placeholders::_1, /*alpha=*/1,
                                 /*inplace=*/false))
                 .call(y));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("log_softmax_grad", LogSoftmaxGradGrad);

}  // namespace one
}  // namespace oneflow
