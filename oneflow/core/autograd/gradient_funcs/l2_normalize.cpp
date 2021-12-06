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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct L2NormalizeCaptureState : public AutoGradCaptureState {
  int64_t axis;
  float epsilon;
  bool requires_grad;
};

class L2Normalize : public OpExprGradFunction<L2NormalizeCaptureState> {
 public:
  Maybe<void> Capture(L2NormalizeCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const L2NormalizeCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> L2Normalize::Capture(L2NormalizeCaptureState* state, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->SaveTensorForBackward(outputs.at(0));  // y
  state->SaveTensorForBackward(outputs.at(1));  // square_x_sum

  auto* interp_ctx = dynamic_cast<const L2NormalizeOpInterpCtx*>(ctx);
  state->axis = interp_ctx->axis;
  state->epsilon = interp_ctx->epsilon;
  return Maybe<void>::Ok();
}

Maybe<void> L2Normalize::Apply(const L2NormalizeCaptureState* state, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(1);
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);
  const auto& y = state->SavedTensors().at(0);
  const auto& square_x_sum = state->SavedTensors().at(1);
  in_grads->at(0) = JUST(
      functional::L2NormalizeGrad(out_grads.at(0), y, square_x_sum, state->axis, state->epsilon));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("l2_normalize", L2Normalize);

}  // namespace one
}  // namespace oneflow
