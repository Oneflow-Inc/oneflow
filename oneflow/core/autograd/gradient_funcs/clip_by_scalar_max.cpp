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

struct ClipByScalarMaxCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Scalar max;
};

class ClipByScalarMax : public OpExprGradFunction<ClipByScalarMaxCaptureState> {
 public:
  Maybe<void> Capture(ClipByScalarMaxCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    state->SaveTensorForBackward(inputs.at(0));

    auto* op_ctx = JUST(ctx->dyn_cast<ClipByScalarMaxOp>());
    if (IsFloatingDataType(inputs.at(0)->dtype()->data_type())) {
      state->max = op_ctx->floating_max();
    } else if (IsIntegralDataType(inputs.at(0)->dtype()->data_type())) {
      state->max = op_ctx->integral_max();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Data type is not floating or integral type.";
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ClipByScalarMaxCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) =
          JUST(functional::ClampGrad(out_grads.at(0), x, /*min=*/NullOpt, state->max));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("clip_by_scalar_max", ClipByScalarMax);

}  // namespace one
}  // namespace oneflow
