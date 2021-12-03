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

enum class ReduceMode : int32_t {
  kMin = 0,
  kMax = 1,
};

struct ReduceDeviceCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad = false;
  size_t mask_index = -1;
  size_t count_index = -1;
};

template<ReduceMode mode>
class ReduceDevice : public OpExprGradFunction<ReduceDeviceCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReduceDeviceCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* interp_ctx = dynamic_cast<const ReduceMaxDeviceStageOpInterpCtx*>(ctx);
    state->axis = interp_ctx->axis;
    state->mask_index = state->SaveTensorForBackward(outputs.at(1));   // mask
    state->count_index = state->SaveTensorForBackward(outputs.at(2));  // count
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReduceDeviceCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 3);
    const auto& mask = state->SavedTensors().at(state->mask_index);
    const auto& count = state->SavedTensors().at(state->count_index);
    in_grads->resize(1);
    if (mode == ReduceMode::kMin) {
      in_grads->at(0) =
          JUST(functional::ReduceMinDeviceStageGrad(out_grads.at(0), mask, count, state->axis));
    } else {
      in_grads->at(0) =
          JUST(functional::ReduceMaxDeviceStageGrad(out_grads.at(0), mask, count, state->axis));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min_device_stage", ReduceDevice<ReduceMode::kMin>);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max_device_stage", ReduceDevice<ReduceMode::kMax>);

struct ReduceGlobalCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad = false;
  bool keepdims = false;
  size_t mask_index = -1;
  size_t device_count_index = -1;
};

template<ReduceMode mode>
class ReduceGlobal : public OpExprGradFunction<ReduceGlobalCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReduceGlobalCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 2);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* interp_ctx = dynamic_cast<const ReduceMaxGlobalStageOpInterpCtx*>(ctx);
    state->axis = interp_ctx->axis;
    state->keepdims = interp_ctx->keepdims;
    state->mask_index = state->SaveTensorForBackward(outputs.at(1));         // mask
    state->device_count_index = state->SaveTensorForBackward(inputs.at(1));  // device_count
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReduceGlobalCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);
    const auto& mask = state->SavedTensors().at(state->mask_index);
    const auto& device_count = state->SavedTensors().at(state->device_count_index);
    in_grads->resize(2);
    if (mode == ReduceMode::kMin) {
      in_grads->at(0) = JUST(functional::ReduceMinGlobalStageGrad(
          out_grads.at(0), mask, device_count, state->axis, state->keepdims));
    } else {
      in_grads->at(0) = JUST(functional::ReduceMaxGlobalStageGrad(
          out_grads.at(0), mask, device_count, state->axis, state->keepdims));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min_global_stage", ReduceGlobal<ReduceMode::kMin>);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max_global_stage", ReduceGlobal<ReduceMode::kMax>);

}  // namespace one
}  // namespace oneflow
