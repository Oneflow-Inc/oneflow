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
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct HardTanhInterpState : public OpExprInterpState {
  bool requires_grad;
  double min_val;
  double max_val;
};

class HardTanh : public OpExprGradFunction<HardTanhInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(HardTanhInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->min_val = JUST(composed_attrs.GetAttr<double>("min_val"));
    ctx->max_val = JUST(composed_attrs.GetAttr<double>("max_val"));
    ctx->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HardTanhInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(0) =
          JUST(functional::HardTanhGrad(y, out_grads.at(0), ctx->min_val, ctx->max_val));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("hardtanh", HardTanh);

}  // namespace one
}  // namespace oneflow
