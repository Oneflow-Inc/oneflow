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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct FusedCrossInteractionInterpState : public AutoGradCaptureState {
  bool x_requires_grad = true;
  bool weight_requires_grad = true;
  bool x0_requires_grad = true;
  bool bias_requires_grad = true;
  size_t x_idx = 0;
  size_t weight_idx = 0;
  size_t x0_idx = 0;
  size_t matmul_result_idx = 0;
};

class FusedCrossInteraction : public OpExprGradFunction<FusedCrossInteractionInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedCrossInteractionInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 4);
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->weight_requires_grad = inputs.at(1)->requires_grad();
    ctx->x_requires_grad = inputs.at(2)->requires_grad();
    ctx->weight_requires_grad = inputs.at(3)->requires_grad();
    ctx->x_idx = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->weight_idx = ctx->SaveTensorForBackward(inputs.at(1));
    ctx->x0_idx = ctx->SaveTensorForBackward(inputs.at(2));
    ctx->matmul_result_idx = ctx->SaveTensorForBackward(outputs.at(1));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedCrossInteractionInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);
    in_grads->resize(4);
    std::shared_ptr<oneflow::one::TensorTuple> grads;
    grads = JUST(functional::FusedCrossInteractionGrad(
        JUST(oneflow::VectorAt(out_grads, 0)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->weight_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x0_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->matmul_result_idx))));
    if (ctx->x_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(oneflow::VectorAt(*grads, 0));
    }
    if (ctx->weight_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 1)) = JUST(oneflow::VectorAt(*grads, 1));
    }
    if (ctx->x0_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 2)) = JUST(oneflow::VectorAt(*grads, 2));
    }
    if (ctx->bias_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 3)) = JUST(oneflow::VectorAt(*grads, 3));
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_cross_interaction", FusedCrossInteraction);

}  // namespace one
}  // namespace oneflow
