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

struct DotCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool y_requires_grad = false;
  size_t x_offset = 0;
  size_t y_offset = 0;
};

class DotGrad : public OpExprGradFunction<DotCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(DotCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    if (ctx->x_requires_grad) { ctx->x_offset = ctx->SaveTensorForBackward(inputs.at(1)); }
    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    if (ctx->y_requires_grad) { ctx->y_offset = ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const DotCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      const auto& x = ctx->SavedTensors().at(ctx->x_offset);
      const auto& results = JUST(functional::Mul(x, out_grads.at(0)));
      in_grads->at(0) = results;
    }

    if (ctx->y_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_offset);
      const auto& results = JUST(functional::Mul(y, out_grads.at(0)));
      in_grads->at(1) = results;
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("dot", DotGrad);

}  // namespace one
}  // namespace oneflow
