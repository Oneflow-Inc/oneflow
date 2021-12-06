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

struct UnfoldTensorCaptureState : public AutoGradCaptureState {
  int32_t dimension = -1;
  int32_t size = -1;
  int32_t step = -1;
  bool requires_grad = false;
};

class UnfoldTensor : public OpExprGradFunction<UnfoldTensorCaptureState> {
 public:
  Maybe<void> Capture(UnfoldTensorCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const UnfoldTensorCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};
Maybe<void> UnfoldTensor::Capture(UnfoldTensorCaptureState* state, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* interp_ctx = dynamic_cast<const UnfoldTensorOpInterpCtx*>(ctx);
  state->dimension = interp_ctx->dimension;
  state->size = interp_ctx->size;
  state->step = interp_ctx->step;
  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> UnfoldTensor::Apply(const UnfoldTensorCaptureState* state, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& in = state->SavedTensors().at(0);
  in_grads->at(0) = JUST(functional::UnfoldTensorGrad(out_grads.at(0), in, state->dimension,
                                                      state->size, state->step));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("unfold_tensor", UnfoldTensor);

}  // namespace one
}  // namespace oneflow
