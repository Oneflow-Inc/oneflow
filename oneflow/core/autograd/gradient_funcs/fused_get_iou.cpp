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
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedGetIouGradCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  float eps = 1e-8;
};

class FusedGetIouGrad : public OpExprGradFunction<FusedGetIouGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedGetIouGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 5);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad() && inputs.at(1)->requires_grad()
                         && inputs.at(4)->requires_grad();
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->eps = JUST(composed_attrs.GetAttr<float>("eps"));
    if (ctx->requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(0));  // w1
      ctx->SaveTensorForBackward(inputs.at(1));  // h1
      ctx->SaveTensorForBackward(inputs.at(2));  // w2
      ctx->SaveTensorForBackward(inputs.at(3));  // h2
      ctx->SaveTensorForBackward(inputs.at(4));  // inter
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedGetIouGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& diou = out_grads.at(0);

    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(saved_tensors.size(), 5);
    const auto& w1 = saved_tensors.at(0);
    const auto& h1 = saved_tensors.at(1);
    const auto& w2 = saved_tensors.at(2);
    const auto& h2 = saved_tensors.at(3);
    const auto& inter = saved_tensors.at(4);

    in_grads->resize(5);
    auto result = JUST(functional::FusedGetIouGrad(diou, w1, h1, w2, h2, inter, ctx->eps));
    CHECK_EQ_OR_RETURN(result->size(), 3);
    if (ctx->requires_grad) {
      in_grads->at(0) = result->at(0);
      in_grads->at(1) = result->at(1);
      in_grads->at(4) = result->at(2);
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_get_iou", FusedGetIouGrad);

}  // namespace one
}  // namespace oneflow
