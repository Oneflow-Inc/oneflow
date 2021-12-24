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

struct FusedSelfAttentionInterpState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  float alpha = 1.0;
};

class FusedSelfAttention : public OpExprGradFunction<FusedSelfAttentionInterpState> {
 public:
  Maybe<void> Capture(FusedSelfAttentionInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->input_requires_grad = inputs.at(0)->requires_grad();
    if (!state->input_requires_grad) { return Maybe<void>::Ok(); }
    auto* interp_ctx = dynamic_cast<const FusedSelfAttentionQueryMulKeyAndValueOp*>(ctx);
    state->alpha = interp_ctx->alpha();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedSelfAttentionInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->input_requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 2);
    in_grads->resize(1);
    const auto& hidden_states = state->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& fused_self_attention_grad =
        JUST(functional::FusedSelfAttentionGrad(out_grads.at(0), out_grads.at(1), hidden_states,
                                                state->alpha));
    in_grads->at(0) = fused_self_attention_grad;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_self_attention_query_mul_key_and_value", FusedSelfAttention);

}  // namespace one
}  // namespace oneflow
