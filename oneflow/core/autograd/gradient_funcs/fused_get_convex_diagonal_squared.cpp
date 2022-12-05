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
struct FusedGetConvexDiagonalSquaredCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
  float eps = 1e-8;
};

class FusedGetConvexDiagonalSquaredGrad
    : public OpExprGradFunction<FusedGetConvexDiagonalSquaredCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedGetConvexDiagonalSquaredCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), INPUT_LEN);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    for (int i = 0; i < INPUT_LEN; i++) {
      ctx->requires_grad.push_back(inputs.at(i)->requires_grad());
    }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->eps = JUST(composed_attrs.GetAttr<float>("eps"));
    for (int i = 0; i < INPUT_LEN; i++) { ctx->SaveTensorForBackward(inputs.at(i)); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedGetConvexDiagonalSquaredCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& c2_diff = out_grads.at(0);

    const auto& b1_x1 = ctx->SavedTensors().at(0);
    const auto& b1_x2 = ctx->SavedTensors().at(1);
    const auto& b2_x1 = ctx->SavedTensors().at(2);
    const auto& b2_x2 = ctx->SavedTensors().at(3);
    const auto& b1_y1 = ctx->SavedTensors().at(4);
    const auto& b1_y2 = ctx->SavedTensors().at(5);
    const auto& b2_y1 = ctx->SavedTensors().at(6);
    const auto& b2_y2 = ctx->SavedTensors().at(7);

    in_grads->resize(INPUT_LEN);
    auto result = JUST(functional::FusedGetConvexDiagonalSquaredGrad(
        c2_diff, b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, ctx->eps));

    CHECK_EQ_OR_RETURN(result->size(), INPUT_LEN);
    for (int i = 0; i < INPUT_LEN; i++) {
      if (ctx->requires_grad[i]) { in_grads->at(i) = result->at(i); }
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_get_convex_diagonal_squared",
                               FusedGetConvexDiagonalSquaredGrad);

}  // namespace one
}  // namespace oneflow
