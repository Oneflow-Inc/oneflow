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

struct FusedGruCellGradCaptureState : public AutoGradCaptureState {
  bool has_bias = true;
  bool hx_needs_grad = true;
};

class FusedGruCellGrad : public OpExprGradFunction<FusedGruCellGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "FusedGruCellGrad::Init forward op expr is null.";
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedGruCellGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    const size_t in_size = inputs.size();
    CHECK_OR_RETURN(in_size == 3 || in_size == 5)
        << "FusedGruCellGrad::Capture(): input tensor size must be 3 or 5";
    ctx->has_bias = in_size == 5;
    ctx->hx_needs_grad = inputs[2]->requires_grad();
    ctx->SaveTensorForBackward(outputs[1]);  // workspace
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedGruCellGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& workspace = ctx->SavedTensors()[0];  // workspace
    const auto& grad_hy = out_grads[0];
    const auto& results =
        JUST(functional::FusedGruCellGrad(grad_hy, workspace, ctx->has_bias, ctx->hx_needs_grad));

    if (ctx->has_bias) {
      in_grads->resize(5);
    } else {
      in_grads->resize(3);
    }
    (*in_grads)[0] = (*results)[0];
    (*in_grads)[1] = (*results)[1];

    if (ctx->hx_needs_grad) { (*in_grads)[2] = (*results)[2]; }

    if (ctx->has_bias) {
      if (ctx->hx_needs_grad) {
        (*in_grads)[3] = (*results)[3];
        (*in_grads)[4] = (*results)[4];
      } else {
        (*in_grads)[3] = (*results)[2];
        (*in_grads)[4] = (*results)[3];
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_gru_cell", FusedGruCellGrad);

}  // namespace one
}  // namespace oneflow
