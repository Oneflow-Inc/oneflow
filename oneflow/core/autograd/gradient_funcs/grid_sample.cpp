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

struct GridSampleInterpState : public AutoGradCaptureState {
  std::string interpolation_mode = "";
  std::string padding_mode = "";
  bool align_corners = false;
  size_t input_index = -1;
  size_t grid_index = -1;
  bool input_requires_grad = false;
  bool grid_requires_grad = false;
  bool requires_grad = false;
};

class GridSample : public OpExprGradFunction<GridSampleInterpState> {
 public:
  Maybe<void> Capture(GridSampleInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->input_requires_grad = inputs.at(0)->requires_grad();
    state->grid_requires_grad = inputs.at(1)->requires_grad();
    state->requires_grad = state->input_requires_grad || state->grid_requires_grad;
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    state->input_index = state->SaveTensorForBackward(inputs.at(0));  // input
    state->grid_index = state->SaveTensorForBackward(inputs.at(1));   // grid

    auto* op_ctx = dynamic_cast<const GridSampleOp*>(ctx);
    state->interpolation_mode = op_ctx->interpolation_mode();
    state->padding_mode = op_ctx->padding_mode();
    state->align_corners = op_ctx->align_corners();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const GridSampleInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);

    const auto& input = state->SavedTensors().at(state->input_index);
    const auto& grid = state->SavedTensors().at(state->grid_index);
    const auto& results =
        JUST(functional::GridSampleGrad(out_grads.at(0), input, grid, state->interpolation_mode,
                                        state->padding_mode, state->align_corners));
    in_grads->resize(2);
    if (state->input_requires_grad) { in_grads->at(0) = results->at(0); }
    if (state->grid_requires_grad) { in_grads->at(1) = results->at(1); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("grid_sample", GridSample);

}  // namespace one
}  // namespace oneflow
