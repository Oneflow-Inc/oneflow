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

struct FusedBiasAddGeluInterpState : public AutoGradCaptureState {
  bool input_requires_grad = true;
  bool bias_requires_grad = true;
  int32_t axis = 1;
};

class FusedBiasAddGelu : public OpExprGradFunction<FusedBiasAddGeluInterpState> {
 public:
  Maybe<void> Capture(FusedBiasAddGeluInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->input_requires_grad = inputs.at(0)->requires_grad();
    state->bias_requires_grad = inputs.at(1)->requires_grad();
    auto* op_ctx = dynamic_cast<const FusedBiasAddGeluOp*>(ctx);
    state->axis = op_ctx->axis();
    if (state->input_requires_grad || state->bias_requires_grad) {
      state->SaveTensorForBackward(inputs.at(0));
      state->SaveTensorForBackward(inputs.at(1));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedBiasAddGeluInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->input_requires_grad && !state->bias_requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const int64_t num_axes = out_grads.at(0)->shape()->NumAxes();
    in_grads->resize(2);
    const auto& a = state->SavedTensors().at(0);
    const auto& b = state->SavedTensors().at(1);
    const std::shared_ptr<oneflow::one::Tensor>& fused_bias_add_gelu_grad =
        JUST(functional::FusedBiasAddGeluGrad(a, b, out_grads.at(0), state->axis));
    if (state->bias_requires_grad) {
      std::vector<int32_t> reduce_axes_vec;
      reduce_axes_vec.reserve(num_axes);
      for (int i = 0; i < num_axes; ++i) {
        if (i != state->axis) { reduce_axes_vec.emplace_back(i); }
      }
      in_grads->at(1) =
          JUST(functional::ReduceSum(fused_bias_add_gelu_grad, reduce_axes_vec, false));
    }
    if (state->input_requires_grad) { in_grads->at(0) = fused_bias_add_gelu_grad; }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_bias_add_gelu", FusedBiasAddGelu);

}  // namespace one
}  // namespace oneflow
