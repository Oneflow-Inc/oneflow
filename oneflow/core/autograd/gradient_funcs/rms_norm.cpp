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

struct RMSNormCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool weight_requires_grad = false;
  int x_index = -1;
  int inv_rms_index = -1;
  int weight_index = -1;
};

class RMSNormGrad : public OpExprGradFunction<RMSNormCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(RMSNormCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const RMSNormCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> RMSNormGrad::Capture(RMSNormCaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  // (x, [weight])
  CHECK_GE_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_LE_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
  // (y, inv_rms)
  CHECK_EQ_OR_RETURN(outputs.size(), 2);  // NOLINT(maybe-need-error-msg)

  // save x
  ctx->x_requires_grad = inputs[0]->requires_grad();
  ctx->x_index = ctx->SaveTensorForBackward(inputs[0]);

  // save weight
  ctx->weight_requires_grad = false;
  if (inputs.size() > 1) {
    ctx->weight_requires_grad = inputs[1]->requires_grad();
    ctx->weight_index = ctx->SaveTensorForBackward(inputs[1]);
  }

  // save inv_rms
  if (ctx->x_requires_grad || ctx->weight_requires_grad) {
    ctx->inv_rms_index = ctx->SaveTensorForBackward(outputs[1]);
  }
  return Maybe<void>::Ok();
}

Maybe<void> RMSNormGrad::Apply(const RMSNormCaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  // (x, inv_rms) or (x, weight, inv_rms)
  const auto& saved_tensors = ctx->SavedTensors();
  CHECK_GE_OR_RETURN(saved_tensors.size(), 2);  // NOLINT(maybe-need-error-msg)
  CHECK_LE_OR_RETURN(saved_tensors.size(), 3);  // NOLINT(maybe-need-error-msg)

  // (dy, inv_rms_diff)
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
  const auto& dy = out_grads[0];
  const auto& x = saved_tensors.at(ctx->x_index);
  const auto& inv_rms = saved_tensors.at(ctx->inv_rms_index);

  // (x_grad, weight_grad)
  in_grads->resize(2);
  if (ctx->x_requires_grad) {
    if (saved_tensors.size() == 3) {
      const auto& weight = saved_tensors.at(ctx->weight_index);
      in_grads->at(0) = JUST(functional::RMSNormGrad(dy, x, inv_rms, weight, /*param_grad*/ false));
    } else {
      in_grads->at(0) =
          JUST(functional::RMSNormGrad(dy, x, inv_rms, /*weight*/ NullOpt, /*param_grad*/ false));
    }
  }
  if (ctx->weight_requires_grad) {
    in_grads->at(1) =
        JUST(functional::RMSNormGrad(dy, x, inv_rms, /*weight*/ NullOpt, /*param_grad*/ true));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("rms_norm", RMSNormGrad);

}  // namespace one
}  // namespace oneflow
