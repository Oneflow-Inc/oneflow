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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {
namespace one {

struct NarrowCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Shape shape;
  int64_t dim;
  int64_t start;
  int64_t length;
};

class Narrow : public OpExprGradFunction<NarrowCaptureState> {
 public:
  Maybe<void> Capture(NarrowCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<NarrowOp>());
    state->dim = op_ctx->dim();
    state->start = op_ctx->start();
    state->length = op_ctx->length();
    if (LazyMode::is_enabled()) {
      state->SaveTensorForBackward(inputs.at(0));
    } else {
      state->shape = *(inputs.at(0)->shape());
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NarrowCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& dy = out_grads.at(0);
    if (state->requires_grad) {
      std::shared_ptr<Tensor> like;
      if (LazyMode::is_enabled()) {
        like = state->SavedTensors().at(0);
      } else {
        like = JUST(functional::Empty(state->shape, dy->dtype(), JUST(dy->device())));
      }
      in_grads->resize(1);
      in_grads->at(0) =
          JUST(functional::NarrowGrad(dy, like, state->dim, state->start, state->length));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("narrow", Narrow);

}  // namespace one
}  // namespace oneflow
