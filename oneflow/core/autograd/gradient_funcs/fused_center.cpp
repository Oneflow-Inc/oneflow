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

const int32_t INPUT_LEN = 8;
struct FusedCenterCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
};

class FusedCenterGrad : public OpExprGradFunction<FusedCenterCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FusedCenterCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), INPUT_LEN);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    for (int i = 0; i < INPUT_LEN; i++) {
      ctx->requires_grad.push_back(inputs.at(i)->requires_grad());
    }
    for (int i = 0; i < INPUT_LEN; i++) { ctx->SaveTensorForBackward(inputs.at(i)); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedCenterCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& rho2_diff = out_grads.at(0);

    const auto& b1_x1 = ctx->SavedTensors().at(0);
    const auto& b1_x2 = ctx->SavedTensors().at(1);
    const auto& b2_x1 = ctx->SavedTensors().at(2);
    const auto& b2_x2 = ctx->SavedTensors().at(3);
    const auto& b1_y1 = ctx->SavedTensors().at(4);
    const auto& b1_y2 = ctx->SavedTensors().at(5);
    const auto& b2_y1 = ctx->SavedTensors().at(6);
    const auto& b2_y2 = ctx->SavedTensors().at(7);

    in_grads->resize(INPUT_LEN);
    auto result = JUST(functional::FusedCenterGrad(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1,
                                                   b2_y2, rho2_diff));

    CHECK_EQ_OR_RETURN(result->size(), INPUT_LEN);
    for (int i = 0; i < INPUT_LEN; i++) {
      if (ctx->requires_grad[i]) { in_grads->at(i) = result->at(i); }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_get_center_dist", FusedCenterGrad);

}  // namespace one
}  // namespace oneflow
