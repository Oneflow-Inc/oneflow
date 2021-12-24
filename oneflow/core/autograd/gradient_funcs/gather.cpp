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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct GatherCaptureState : public AutoGradCaptureState {
  int64_t axis;
  bool requires_grad;
};

class Gather : public OpExprGradFunction<GatherCaptureState> {
 public:
  Maybe<void> Capture(GatherCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const GatherCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Gather::Capture(GatherCaptureState* state, const TensorTuple& inputs,
                            const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->SaveTensorForBackward(inputs.at(0));
  state->SaveTensorForBackward(inputs.at(1));

  auto* interp_ctx = dynamic_cast<const GatherOp*>(ctx);
  state->axis = interp_ctx->axis();
  return Maybe<void>::Ok();
}

Maybe<void> Gather::Apply(const GatherCaptureState* state, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& x = state->SavedTensors().at(0);
  const auto& indices = state->SavedTensors().at(1);
  in_grads->at(0) =
      JUST(functional::UnsortedSegmentSumLike(out_grads.at(0), indices, x, state->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("gather", Gather);

}  // namespace one
}  // namespace oneflow
