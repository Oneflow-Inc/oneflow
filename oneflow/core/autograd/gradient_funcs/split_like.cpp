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
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SplitLikeCaptureState : public AutoGradCaptureState {
  int64_t axis;
  bool requires_grad;
};

class SplitLike : public OpExprGradFunction<SplitLikeCaptureState> {
 public:
  Maybe<void> Capture(SplitLikeCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const SplitLikeCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> SplitLike::Capture(SplitLikeCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), outputs.size() + 1);
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  auto* interp_ctx = dynamic_cast<const SplitLikeOpInterpCtx*>(ctx);
  state->axis = interp_ctx->axis();
  for (int i = 0; i < outputs.size(); ++i) { state->SaveTensorForBackward(outputs.at(i)); }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Apply(const SplitLikeCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  in_grads->resize(1);
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  const auto& saved_tensors = state->SavedTensors();
  TensorTuple inputs;
  inputs.reserve(out_grads.size());
  for (int i = 0; i < out_grads.size(); ++i) {
    const auto& out_grad_i = out_grads.at(i);
    if (out_grad_i.get()) {
      inputs.emplace_back(out_grad_i);
    } else {
      const auto& zero_grad = JUST(functional::ZerosLike(saved_tensors.at(i)));
      inputs.emplace_back(zero_grad);
    }
  }
  in_grads->at(0) = JUST(functional::Concat(inputs, state->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("split_like", SplitLike);

}  // namespace one
}  // namespace oneflow
