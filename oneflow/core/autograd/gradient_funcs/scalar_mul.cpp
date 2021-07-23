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

struct ScalarMulInterpState : public OpExprInterpState {
  bool requires_grad;
  functional::Scalar operand;
};

class ScalarMul : public OpExprGradFunction<ScalarMulInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarMulInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = functional::Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = functional::Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarMulInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      in_grads->at(0) = JUST(functional::ScalarMul(out_grads.at(0), ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_mul", ScalarMul);

}  // namespace one
}  // namespace oneflow
