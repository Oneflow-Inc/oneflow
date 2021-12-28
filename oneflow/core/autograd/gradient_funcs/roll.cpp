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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct RollCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> shifts;
  std::vector<int32_t> dims;
  bool requires_grad = false;
};

class Roll : public OpExprGradFunction<RollCaptureState> {
 public:
  Maybe<void> Capture(RollCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const RollCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Roll::Capture(RollCaptureState* state, const TensorTuple& inputs,
                          const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* op_ctx = dynamic_cast<const RollOp*>(ctx);
  state->shifts = op_ctx->shifts();
  state->dims = op_ctx->dims();
  return Maybe<void>::Ok();
}

Maybe<void> Roll::Apply(const RollCaptureState* state, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  std::vector<int32_t> new_shifts;
  new_shifts.resize(state->shifts.size());
  for (int i = 0; i < new_shifts.size(); ++i) { new_shifts[i] = -state->shifts[i]; }

  in_grads->at(0) = JUST(functional::Roll(out_grads.at(0), new_shifts, state->dims));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("roll", Roll);

}  // namespace one
}  // namespace oneflow
