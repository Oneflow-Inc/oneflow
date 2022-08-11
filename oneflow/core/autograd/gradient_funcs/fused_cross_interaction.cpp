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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct FusedCrossFeatureInteractionInterpState : public AutoGradCaptureState {
  bool x_requires_grad = true;
  bool weight_requires_grad = true;
  bool x0_requires_grad = true;
  bool bias_requires_grad = true;
  size_t x_idx = 0;
  size_t bias_idx = 0;
  size_t weight_idx = 0;
  size_t x0_idx = 0;
  size_t matmul_result_idx = 0;
  std::string interaction_mode;
};

class FusedCrossFeatureInteraction
    : public OpExprGradFunction<FusedCrossFeatureInteractionInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be None. ";
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedCrossFeatureInteractionInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 4) << "Input size should be equal to 4. ";
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->interaction_mode = JUST(composed_attrs.GetAttr<std::string>("interaction_mode"));
    ctx->x_requires_grad = JUST(oneflow::VectorAt(inputs, 0))->requires_grad();
    ctx->weight_requires_grad = JUST(oneflow::VectorAt(inputs, 1))->requires_grad();
    ctx->x_requires_grad = JUST(oneflow::VectorAt(inputs, 2))->requires_grad();
    ctx->weight_requires_grad = JUST(oneflow::VectorAt(inputs, 3))->requires_grad();
    ctx->x_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 0)));
    ctx->weight_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 1)));
    ctx->x0_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 2)));
    if (ctx->interaction_mode == "matrix") {
      ctx->bias_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 3)));
    }
    ctx->matmul_result_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(outputs, 1)));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedCrossFeatureInteractionInterpState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 2) << "Out grads size should be equal to 2. ";
    std::shared_ptr<oneflow::one::TensorTuple> grads;
    in_grads->resize(4);
    if (ctx->interaction_mode == "vector") {
      grads = JUST(functional::FusedCrossFeatureInteractionV1Grad(
          JUST(oneflow::VectorAt(out_grads, 0)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->weight_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x0_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->matmul_result_idx))));
    } else if (ctx->interaction_mode == "matrix") {
      grads = JUST(functional::FusedCrossFeatureInteractionV2Grad(
          JUST(oneflow::VectorAt(out_grads, 0)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->weight_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->bias_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->x0_idx)),
          JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->matmul_result_idx))));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Interaction mode only support `vector` and `matrix`. ";
    }

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

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_cross_feature_interaction", FusedCrossFeatureInteraction);

}  // namespace one
}  // namespace oneflow
