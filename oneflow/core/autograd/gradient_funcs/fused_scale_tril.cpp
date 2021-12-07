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
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedScaleTrilState : public AutoGradCaptureState {
  bool requires_grad;
  int64_t diagonal;
  double floating_scale_value;
  int64_t integer_scale_value;
  bool is_floating_scale_value;
};

class FusedScaleTril : public OpExprGradFunction<FusedScaleTrilState> {
 public:
  Maybe<void> Capture(FusedScaleTrilState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const FusedScaleTrilState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> FusedScaleTril::Capture(FusedScaleTrilState* state, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  auto* interp_ctx = dynamic_cast<const FusedScaleTrilOpInterpCtx*>(ctx);
  state->diagonal = interp_ctx->diagonal();
  state->floating_scale_value = interp_ctx->floating_scale_value();
  state->integer_scale_value = interp_ctx->integer_scale_value();
  state->is_floating_scale_value = interp_ctx->is_floating_scale_value();
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleTril::Apply(const FusedScaleTrilState* state, const TensorTuple& out_grads,
                                  TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  Scalar scale;
  if (state->is_floating_scale_value) {
    scale = state->floating_scale_value;
  } else {
    scale = state->integer_scale_value;
  }
  (*in_grads)[0] = JUST(functional::FusedScaleTril(out_grads[0], state->diagonal, 0, scale));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_tril", FusedScaleTril);

}  // namespace one
}  // namespace oneflow
