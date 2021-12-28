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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct RoiAlignCaptureState : public AutoGradCaptureState {
  float spatial_scale = 1.0;
  int32_t pooled_h = 0;
  int32_t pooled_w = 0;
  int32_t sampling_ratio = -1;
  bool aligned = false;
  bool requires_grad = false;
};

class RoiAlign : public OpExprGradFunction<RoiAlignCaptureState> {
 public:
  Maybe<void> Capture(RoiAlignCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    state->SaveTensorForBackward(inputs.at(0));
    state->SaveTensorForBackward(inputs.at(1));

    const auto* op_ctx = JUST(ctx->dyn_cast<RoiAlignOp>());
    state->spatial_scale = op_ctx->spatial_scale();
    state->pooled_h = op_ctx->pooled_h();
    state->pooled_w = op_ctx->pooled_w();
    state->sampling_ratio = op_ctx->sampling_ratio();
    state->aligned = op_ctx->aligned();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const RoiAlignCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    const auto& x_like = state->SavedTensors().at(0);
    const auto& rois = state->SavedTensors().at(1);
    in_grads->at(0) = JUST(functional::RoiAlignGrad(
        out_grads.at(0), x_like, rois, state->spatial_scale, state->pooled_h, state->pooled_w,
        state->sampling_ratio, state->aligned));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("roi_align", RoiAlign);

}  // namespace one
}  // namespace oneflow
