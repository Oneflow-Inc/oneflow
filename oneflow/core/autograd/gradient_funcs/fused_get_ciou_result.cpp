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

struct FusedGetCiouResultGradCaptureState : public AutoGradCaptureState {
  bool v_requires_grad = false;
  bool iou_requires_grad = false;
  bool rho2_requires_grad = false;
  bool c2_requires_grad = false;
  float eps = 0.0;
};

class FusedGetCiouResultGrad : public OpExprGradFunction<FusedGetCiouResultGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedGetCiouResultGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 4);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->v_requires_grad = inputs.at(0)->requires_grad();
    ctx->iou_requires_grad = inputs.at(1)->requires_grad();
    ctx->rho2_requires_grad = inputs.at(2)->requires_grad();
    ctx->c2_requires_grad = inputs.at(3)->requires_grad();
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->eps = JUST(composed_attrs.GetAttr<float>("eps"));
    if (ctx->v_requires_grad || ctx->iou_requires_grad || ctx->rho2_requires_grad
        || ctx->c2_requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(0));  // v
      ctx->SaveTensorForBackward(inputs.at(1));  // iou
      ctx->SaveTensorForBackward(inputs.at(2));  // rho2
      ctx->SaveTensorForBackward(inputs.at(3));  // c2
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedGetCiouResultGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& dy = out_grads.at(0);

    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(saved_tensors.size(), 4);
    const auto& v = saved_tensors.at(0);
    const auto& iou = saved_tensors.at(1);
    const auto& rho2 = saved_tensors.at(2);
    const auto& c2 = saved_tensors.at(3);

    in_grads->resize(4);
    auto result = JUST(functional::FusedGetCiouResultGrad(dy, v, iou, rho2, c2, ctx->eps));
    CHECK_EQ_OR_RETURN(result->size(), 4);
    if (ctx->v_requires_grad) { in_grads->at(0) = result->at(0); }
    if (ctx->iou_requires_grad) { in_grads->at(1) = result->at(1); }
    if (ctx->rho2_requires_grad) { in_grads->at(2) = result->at(2); }
    if (ctx->c2_requires_grad) { in_grads->at(3) = result->at(3); }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_get_ciou_result", FusedGetCiouResultGrad);

}  // namespace one
}  // namespace oneflow
