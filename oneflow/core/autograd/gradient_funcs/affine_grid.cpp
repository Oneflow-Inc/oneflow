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

struct AffineGridInterpState : public AutoGradCaptureState {
  Shape size;
  bool align_corners = false;
  bool requires_grad = false;
};

class AffineGrid : public OpExprGradFunction<AffineGridInterpState> {
 public:
  Maybe<void> Capture(AffineGridInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();  // theta
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* interp_ctx = dynamic_cast<const AffineGridOpInterpCtx*>(ctx);
    state->size = interp_ctx->size();
    state->align_corners = interp_ctx->align_corners();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const AffineGridInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    in_grads->at(0) =
        JUST(functional::AffineGridGrad(out_grads.at(0), state->size, state->align_corners));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("affine_grid", AffineGrid);

}  // namespace one
}  // namespace oneflow
