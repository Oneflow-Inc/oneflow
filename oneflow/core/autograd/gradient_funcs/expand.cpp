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
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ExpandCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> logical_out_shape;
  std::vector<int32_t> logical_expand_shape;
  bool requires_grad;
};

class Expand : public OpExprGradFunction<ExpandCaptureState> {
 public:
  Maybe<void> Capture(ExpandCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const ExpandCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Expand::Capture(ExpandCaptureState* state, const TensorTuple& inputs,
                            const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* interp_ctx = dynamic_cast<const ExpandOpInterpCtx*>(ctx);
  state->logical_out_shape = interp_ctx->logical_in_shape();
  state->logical_expand_shape = interp_ctx->logical_expand_shape();
  return Maybe<void>::Ok();
}

Maybe<void> Expand::Apply(const ExpandCaptureState* state, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->at(0) = JUST(functional::ExpandGrad(out_grads.at(0), state->logical_out_shape,
                                                state->logical_expand_shape));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("expand", Expand);

}  // namespace one
}  // namespace oneflow
