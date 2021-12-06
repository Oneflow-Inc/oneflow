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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ClipByScalarMinCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Scalar min;
};

class ClipByScalarMin : public OpExprGradFunction<ClipByScalarMinCaptureState> {
 public:
  Maybe<void> Capture(ClipByScalarMinCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    state->SaveTensorForBackward(inputs.at(0));

    auto* interp_ctx = dynamic_cast<const ClipByScalarMinOpInterpCtx*>(ctx);
    if (IsFloatingDataType(inputs.at(0)->dtype()->data_type())) {
      state->min = interp_ctx->floating_min;
    } else if (IsIntegralDataType(inputs.at(0)->dtype()->data_type())) {
      state->min = interp_ctx->integral_min;
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Data type is not floating or integral type.";
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ClipByScalarMinCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::ClampGrad(out_grads.at(0), x, state->min,
                                                   /*max=*/NullOpt));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("clip_by_scalar_min", ClipByScalarMin);

}  // namespace one
}  // namespace oneflow
