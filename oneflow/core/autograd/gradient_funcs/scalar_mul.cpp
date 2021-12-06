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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ScalarMulCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Scalar operand;
};

class ScalarMul : public OpExprGradFunction<ScalarMulCaptureState> {
 public:
  Maybe<void> Capture(ScalarMulCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* interp_ctx = dynamic_cast<const ScalarMulOpInterpCtx*>(ctx);
    bool has_float_operand = interp_ctx->has_float_operand;
    if (has_float_operand) {
      state->operand = Scalar(interp_ctx->float_operand);
    } else {
      state->operand = Scalar(interp_ctx->int_operand);
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarMulCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      in_grads->at(0) = JUST(functional::ScalarMul(out_grads.at(0), state->operand, false));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_mul", ScalarMul);

}  // namespace one
}  // namespace oneflow
