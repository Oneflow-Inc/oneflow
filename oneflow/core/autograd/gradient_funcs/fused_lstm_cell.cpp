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
  bool has_bias = true;
  bool need_grad_cx = true;
};

class FusedLstmCellGrad : public OpExprGradFunction<FusedLstmCellGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "FusedLstmCellGrad::Init forward op expr is null.";
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedLstmCellGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    const size_t in_size = inputs.size();
    CHECK_OR_RETURN(in_size == 3 || in_size == 5)
        << "FusedLstmCellGrad::Capture(): input tensor size must be 3 or 5";
    ctx->has_bias = in_size == 5;
    ctx->need_grad_cx = inputs[2]->requires_grad();
    ctx->SaveTensorForBackward(inputs[2]);   // cx
    ctx->SaveTensorForBackward(outputs[1]);  // cy
    ctx->SaveTensorForBackward(outputs[2]);  // workspace
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedLstmCellGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& cx = ctx->SavedTensors()[0];         // cx
    const auto& cy = ctx->SavedTensors()[1];         // cy
    const auto& workspace = ctx->SavedTensors()[2];  // workspace

    const auto& grad_hy = out_grads[0];
    const auto& grad_cy = out_grads[1];

    const auto& results = JUST(functional::FusedLstmCellGrad(grad_hy, grad_cy, cx, cy, workspace,
                                                             ctx->need_grad_cx, ctx->has_bias));

    if (ctx->has_bias) {
      in_grads->resize(5);
    } else {
      in_grads->resize(3);
    }
    (*in_grads)[0] = (*results)[0];
    (*in_grads)[1] = (*results)[0];

    if (ctx->need_grad_cx) { (*in_grads)[2] = (*results)[1]; }

    if (ctx->has_bias) {
      if (ctx->need_grad_cx) {
        (*in_grads)[3] = (*results)[2];
        (*in_grads)[4] = (*results)[2];
      } else {
        (*in_grads)[3] = (*results)[1];
        (*in_grads)[4] = (*results)[1];
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_lstm_cell", FusedLstmCellGrad);

}  // namespace one
}  // namespace oneflow
