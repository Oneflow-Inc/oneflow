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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedScaleMaskSoftmaxDropoutInterState : public AutoGradCaptureState {
  bool input_requires_grad = true;
  float scale = 1.0;
  float dropout_scale = 1.0;
};

class FusedScaleMaskSoftmaxDropout
    : public OpExprGradFunction<FusedScaleMaskSoftmaxDropoutInterState> {
 public:
  Maybe<void> Capture(FusedScaleMaskSoftmaxDropoutInterState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const FusedScaleMaskSoftmaxDropoutInterState* state,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;
};

Maybe<void> FusedScaleMaskSoftmaxDropout::Capture(FusedScaleMaskSoftmaxDropoutInterState* state,
                                                  const TensorTuple& inputs,
                                                  const TensorTuple& outputs,
                                                  const OpBase* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // input, mask, dropout_mask
  state->input_requires_grad = inputs.at(0)->requires_grad();

  if (!state->input_requires_grad) { return Maybe<void>::Ok(); }
  auto* op_ctx = dynamic_cast<const FusedScaleMaskSoftmaxDropoutOp*>(ctx);
  state->scale = op_ctx->scale_value();
  state->dropout_scale = op_ctx->dropout_scale_value();

  state->SaveTensorForBackward(inputs.at(1));   // mask
  state->SaveTensorForBackward(inputs.at(2));   // dropout_mask
  state->SaveTensorForBackward(outputs.at(1));  // softmax_y
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmaxDropout::Apply(const FusedScaleMaskSoftmaxDropoutInterState* state,
                                                const TensorTuple& out_grads,
                                                TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // dy, d_softmax_y
  if (!state->input_requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(3);  // input, mask, dropout_mask

  const std::shared_ptr<oneflow::one::Tensor>& mask = state->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& dropout_mask = state->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& softmax_y = state->SavedTensors().at(2);
  const std::shared_ptr<oneflow::one::Tensor>& input_grad =
      JUST(functional::FusedScaleMaskSoftmaxDropoutGrad(
          softmax_y, out_grads.at(0), mask, dropout_mask, state->scale, state->dropout_scale));

  in_grads->at(0) = input_grad;
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_mask_softmax_dropout", FusedScaleMaskSoftmaxDropout);

}  // namespace one
}  // namespace oneflow
