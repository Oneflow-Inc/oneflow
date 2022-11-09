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
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct ScalarDivCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  Scalar operand;
};

class ScalarDiv : public OpExprGradFunction<ScalarDivCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarDivCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarDivCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(1);
    if (ctx->requires_grad) {
      JUST(VectorAt(*in_grads, 0)) =
          JUST(functional::ScalarDiv(JUST(VectorAt(out_grads, 0)), ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_div", ScalarDiv);

}  // namespace one
}  // namespace oneflow
