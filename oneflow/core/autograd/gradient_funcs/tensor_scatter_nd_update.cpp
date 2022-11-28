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

struct TensorScatterNdUpdateCaptureState : public AutoGradCaptureState {
  bool tensor_requires_grad = false;
  bool update_requires_grad = false;
};

class TensorScatterNdUpdate : public OpExprGradFunction<TensorScatterNdUpdateCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(TensorScatterNdUpdateCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->tensor_requires_grad = inputs.at(0)->requires_grad();
    ctx->update_requires_grad = inputs.at(2)->requires_grad();
    if (ctx->update_requires_grad || ctx->tensor_requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(1));  // indices
    }
    if (ctx->tensor_requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(2));  // update: only use meta information
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const TensorScatterNdUpdateCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(3);
    if (ctx->update_requires_grad) {
      const auto& indices = ctx->SavedTensors().at(0);
      in_grads->at(2) = JUST(functional::GatherNd(out_grads.at(0), indices));
    }
    if (ctx->tensor_requires_grad) {
      const auto& indices = ctx->SavedTensors().at(0);
      const auto& update = ctx->SavedTensors().at(1);
      const auto& temp = JUST(functional::ZerosLike(update));
      in_grads->at(0) = JUST(
          functional::TensorScatterNdUpdate(out_grads.at(0), indices, temp, /*inplace=*/false));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tensor_scatter_nd_update", TensorScatterNdUpdate);

}  // namespace one
}  // namespace oneflow
