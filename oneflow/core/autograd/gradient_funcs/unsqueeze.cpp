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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {
namespace one {

struct UnsqueezeCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Shape shape;
};

class Unsqueeze : public OpExprGradFunction<UnsqueezeCaptureState> {
 public:
  Maybe<void> Capture(UnsqueezeCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const UnsqueezeCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};
Maybe<void> Unsqueeze::Capture(UnsqueezeCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  if (LazyMode::is_enabled()) {
    state->SaveTensorForBackward(inputs.at(0));
  } else {
    state->shape = *(inputs.at(0)->shape());
  }
  return Maybe<void>::Ok();
}

Maybe<void> Unsqueeze::Apply(const UnsqueezeCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  in_grads->resize(1);
  if (LazyMode::is_enabled()) {
    const auto& like = state->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::ReshapeLike(out_grads.at(0), like));
  } else {
    in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), state->shape));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("expand_dims", Unsqueeze);

}  // namespace one
}  // namespace oneflow
