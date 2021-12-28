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

struct FusedScaleMaskSoftmaxInterState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  float scale = 1.0;
};

class FusedScaleMaskSoftmax : public OpExprGradFunction<FusedScaleMaskSoftmaxInterState> {
 public:
  Maybe<void> Capture(FusedScaleMaskSoftmaxInterState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const FusedScaleMaskSoftmaxInterState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> FusedScaleMaskSoftmax::Capture(FusedScaleMaskSoftmaxInterState* state,
                                           const TensorTuple& inputs, const TensorTuple& outputs,
                                           const OpBase* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);  // input, mask
  state->input_requires_grad = inputs.at(0)->requires_grad();

  if (!state->input_requires_grad) { return Maybe<void>::Ok(); }
  auto* op_ctx = dynamic_cast<const FusedScaleMaskSoftmaxOp*>(ctx);
  state->scale = op_ctx->scale_value();

  state->SaveTensorForBackward(inputs.at(1));   // save mask
  state->SaveTensorForBackward(outputs.at(0));  // save y, ie. softmax result
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmax::Apply(const FusedScaleMaskSoftmaxInterState* state,
                                         const TensorTuple& out_grads,
                                         TensorTuple* in_grads) const {
  if (!state->input_requires_grad) { return Maybe<void>::Ok(); }

  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // dy
  in_grads->resize(2);                      // input, mask

  const std::shared_ptr<oneflow::one::Tensor>& mask = state->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& y = state->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& fused_scale_mask_softmax_grad =
      JUST(functional::FusedScaleMaskSoftmaxGrad(y, out_grads.at(0), mask, state->scale));

  in_grads->at(0) = fused_scale_mask_softmax_grad;
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_mask_softmax", FusedScaleMaskSoftmax);

}  // namespace one
}  // namespace oneflow
