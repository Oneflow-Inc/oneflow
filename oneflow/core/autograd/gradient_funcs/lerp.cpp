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
#include <vector>
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

const int32_t INPUT_LEN = 3;
struct LerpCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
};

class LerpGrad : public OpExprGradFunction<LerpCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(LerpCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), INPUT_LEN);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);

    for (int i = 0; i < INPUT_LEN; i++) {
      ctx->requires_grad.push_back(inputs.at(i)->requires_grad());
    }
    for (int i = 0; i < INPUT_LEN; i++) { ctx->SaveTensorForBackward(inputs.at(i)); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LerpCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& out_diff = out_grads.at(0);

    const auto& start = ctx->SavedTensors().at(0);
    const auto& end = ctx->SavedTensors().at(1);
    const auto& weight = ctx->SavedTensors().at(2);

    auto result = JUST(functional::LerpGrad(start, end, weight, out_diff));
    CHECK_EQ_OR_RETURN(result->size(), INPUT_LEN);

    in_grads->resize(INPUT_LEN);
    for (int i = 0; i < INPUT_LEN; i++) {
      if (ctx->requires_grad[i]) { in_grads->at(i) = result->at(i); }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("lerp", LerpGrad);

}  // namespace one
}  // namespace oneflow
