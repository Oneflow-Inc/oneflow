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

struct FusedFastGeluMulGradCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
};

class FusedFastGeluMulGrad : public OpExprGradFunction<FusedFastGeluMulGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FusedFastGeluMulGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // (in, multiplier)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // (out,)
    ctx->requires_grad = inputs.at(0)->requires_grad() || inputs.at(1)->requires_grad();
    if (ctx->requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(0));  // in
      ctx->SaveTensorForBackward(inputs.at(1));  // multiplier
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedFastGeluMulGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& out_diff = out_grads.at(0);

    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(saved_tensors.size(), 2);
    const auto& in = saved_tensors.at(0);
    const auto& multiplier = saved_tensors.at(1);

    in_grads->resize(2);  // (in_diff, multiplier_diff)
    auto result = JUST(functional::FusedFastGeluMulGrad(out_diff, in, multiplier));
    CHECK_EQ_OR_RETURN(result->size(), 2);
    in_grads->at(0) = result->at(0);
    in_grads->at(1) = result->at(1);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_fast_gelu_mul", FusedFastGeluMulGrad);

}  // namespace one
}  // namespace oneflow
