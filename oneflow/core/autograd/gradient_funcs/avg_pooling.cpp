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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct AvgPoolingCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  size_t input_index;
  size_t output_index;

  std::string data_format;
  std::vector<int32_t> padding;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};

class AvgPoolingNdGrad : public OpExprGradFunction<AvgPoolingCaptureState> {
 public:
  virtual ~AvgPoolingNdGrad() = default;
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(AvgPoolingCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const AvgPoolingCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

};

Maybe<void> AvgPoolingNdGrad::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> AvgPoolingNdGrad::Capture(AvgPoolingCaptureState* state, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  state->output_index = state->SaveTensorForBackward(outputs.at(0));

  auto* interp_ctx = dynamic_cast<const AvgPool3DOpInterpCtx*>(ctx);
  state->data_format = interp_ctx->data_format;
  state->padding = interp_ctx->padding;
  state->kernel_size = interp_ctx->kernel_size;
  state->stride = interp_ctx->stride;
  state->ceil_mode = interp_ctx->ceil_mode;
  state->count_include_pad = interp_ctx->count_include_pad;
  state->divisor_override = interp_ctx->divisor_override;

  return Maybe<void>::Ok();
}

Maybe<void> AvgPoolingNdGrad::Apply(const AvgPoolingCaptureState* state, const TensorTuple& out_grads,
                                    TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  int32_t ndims = state->kernel_size.size();
  const auto& input = state->SavedTensors().at(state->input_index);
  const auto& output = state->SavedTensors().at(state->output_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::AvgPoolingNdGrad(
      input, output, out_grads.at(0), ndims, state->data_format, state->padding, state->kernel_size,
      state->stride, state->ceil_mode, state->count_include_pad, state->divisor_override));

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_EXPR_GRAD_FUNCTION("avgpool_1d", AvgPoolingNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("avgpool_2d", AvgPoolingNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("avgpool_3d", AvgPoolingNdGrad);

}  // namespace one
}  // namespace oneflow
