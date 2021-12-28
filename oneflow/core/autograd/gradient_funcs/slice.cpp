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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SliceCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class Slice : public OpExprGradFunction<SliceCaptureState> {
 public:
  Maybe<void> Capture(SliceCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<SliceOp>());
    state->start = op_ctx->start();
    state->stop = op_ctx->stop();
    state->step = op_ctx->step();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& like = state->SavedTensors().at(0);

    in_grads->resize(1);
    in_grads->at(0) =
        JUST(functional::SliceGrad(out_grads.at(0), like, state->start, state->stop, state->step));
    return Maybe<void>::Ok();
  }
};

struct SliceUpdateCaptureState : public AutoGradCaptureState {
  bool requires_grad_x;
  bool requires_grad_update;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class SliceUpdate : public OpExprGradFunction<SliceUpdateCaptureState> {
 public:
  Maybe<void> Capture(SliceUpdateCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad_x = inputs.at(0)->requires_grad();
    state->requires_grad_update = inputs.at(1)->requires_grad();
    if (!state->requires_grad_x && !state->requires_grad_update) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<SliceUpdateOp>());
    state->start = op_ctx->start();
    state->stop = op_ctx->stop();
    state->step = op_ctx->step();

    if (state->requires_grad_x) { state->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceUpdateCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (state->requires_grad_x) {
      const auto& update = state->SavedTensors().at(0);
      const auto& temp = JUST(functional::ZerosLike(update));
      in_grads->at(0) = JUST(functional::SliceUpdate(out_grads.at(0), temp, state->start,
                                                     state->stop, state->step, /*inplace=*/false));
    }
    if (state->requires_grad_update) {
      in_grads->at(1) =
          JUST(functional::Slice(out_grads.at(0), state->start, state->stop, state->step));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("slice", Slice);
REGISTER_OP_EXPR_GRAD_FUNCTION("slice_update", SliceUpdate);

}  // namespace one
}  // namespace oneflow
