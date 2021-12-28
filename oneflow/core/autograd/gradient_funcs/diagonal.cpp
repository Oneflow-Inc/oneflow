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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct DiagonalInterpState : public AutoGradCaptureState {
  bool requires_grad = false;
  int32_t offset = 0;
};

class Diagonal : public OpExprGradFunction<DiagonalInterpState> {
 public:
  Maybe<void> Capture(DiagonalInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const DiagonalInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Diagonal::Capture(DiagonalInterpState* state, const TensorTuple& inputs,
                              const TensorTuple& outputs, const OpBase* ctx) const {
  CHECK_EQ_OR_RETURN(outputs.size(), 1);
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  const auto* op_ctx = dynamic_cast<const DiagonalOp*>(ctx);
  state->offset = JUST(op_ctx->GetAttr<int32_t>("offset"));
  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Diagonal::Apply(const DiagonalInterpState* state, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(2);
  if (state->requires_grad) {
    const auto& x = state->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::DiagonalGrad(out_grads.at(0), x, state->offset));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("diagonal", Diagonal);

}  // namespace one
}  // namespace oneflow
