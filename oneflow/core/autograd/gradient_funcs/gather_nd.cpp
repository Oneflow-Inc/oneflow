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

struct GatherNdCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class GatherNd : public OpExprGradFunction<GatherNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(GatherNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (ctx->requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(0));  // params
      ctx->SaveTensorForBackward(inputs.at(1));  // indices
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const GatherNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);
    if (ctx->requires_grad) {
      const auto& params = ctx->SavedTensors().at(0);
      const auto& indices = ctx->SavedTensors().at(1);
      in_grads->at(0) = JUST(functional::ScatterNdLike(params, out_grads.at(0), indices));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("gather_nd", GatherNd);

}  // namespace one
}  // namespace oneflow
