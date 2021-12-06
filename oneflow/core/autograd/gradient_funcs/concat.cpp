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

struct ConcatCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
  int64_t axis;
  int64_t input_num;
};

class Concat : public OpExprGradFunction<ConcatCaptureState> {
 public:
  Maybe<void> Capture(ConcatCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const ConcatCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Concat::Capture(ConcatCaptureState* state, const TensorTuple& inputs,
                            const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad.resize(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    state->requires_grad[i] = inputs.at(i)->requires_grad();
  }

  auto* interp_ctx = dynamic_cast<const ConcatOpInterpCtx*>(ctx);
  state->axis = interp_ctx->axis;
  for (const auto& input : inputs) { state->SaveTensorForBackward(input); }
  state->input_num = inputs.size();
  return Maybe<void>::Ok();
}

Maybe<void> Concat::Apply(const ConcatCaptureState* state, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(state->input_num);
  TensorTuple like(state->input_num);
  for (int i = 0; i < state->input_num; ++i) { like[i] = state->SavedTensors().at(i); }
  const auto& results = JUST(functional::SplitLike(out_grads.at(0), like, state->axis));
  CHECK_EQ_OR_RETURN(results->size(), state->input_num);

  for (int i = 0; i < state->input_num; ++i)
    if (state->requires_grad.at(i)) { in_grads->at(i) = results->at(i); }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("concat", Concat);

}  // namespace one
}  // namespace oneflow
