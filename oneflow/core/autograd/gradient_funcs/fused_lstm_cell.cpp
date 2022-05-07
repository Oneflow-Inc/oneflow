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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedLstmCellGradCaptureState : public AutoGradCaptureState {
  bool has_bias;
};

class FusedLstmCellGrad : public OpExprGradFunction<FusedLstmCellGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedLstmCellGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    if (inputs.size() == 3) {
      ctx->has_bias = false;
    } else {
      CHECK_EQ_OR_RETURN(inputs.size(), 5);
      ctx->has_bias = true;
    }
    ctx->SaveTensorForBackward(inputs.at(2));   // cx
    ctx->SaveTensorForBackward(outputs.at(1));  // cy
    ctx->SaveTensorForBackward(outputs.at(2));  // workspace
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedLstmCellGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& cx = ctx->SavedTensors().at(0);         // cx
    const auto& cy = ctx->SavedTensors().at(1);         // cy
    const auto& workspace = ctx->SavedTensors().at(2);  // workspace

    const auto& grad_hy = out_grads.at(0);
    const auto& grad_cy = out_grads.at(1);

    const auto& results =
        JUST(functional::FusedLstmCellGrad(grad_hy, grad_cy, cx, cy, workspace, ctx->has_bias));

    if (ctx->has_bias) {
      CHECK_EQ_OR_RETURN(results->size(), 5);
      in_grads->resize(5);
    } else {
      CHECK_EQ_OR_RETURN(results->size(), 3);
      in_grads->resize(3);
    }
    for (int i = 0; i < results->size(); ++i) { in_grads->at(i) = results->at(i); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_lstm_cell", FusedLstmCellGrad);

}  // namespace one
}  // namespace oneflow
