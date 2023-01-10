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

struct ScalarPowCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Scalar operand;
};

class ScalarPow : public OpExprGradFunction<ScalarPowCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarPowCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarPowCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      in_grads->at(0) = JUST(functional::ScalarPowGrad(x, out_grads.at(0), ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_pow", ScalarPow);

class ScalarReversePow : public OpExprGradFunction<ScalarPowCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarPowCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs[0]->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    ctx->SaveTensorForBackward(inputs[0]);
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarPowCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors()[0];
    in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] = JUST(functional::ScalarReversePowGrad(x, out_grads[0], ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_reverse_pow", ScalarReversePow);

}  // namespace one
}  // namespace oneflow
