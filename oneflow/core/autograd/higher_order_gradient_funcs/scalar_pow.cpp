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
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct ScalarPowGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  Scalar operand;
};

class ScalarPowGradGrad : public OpExprGradFunction<ScalarPowGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarPowGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!(ctx->x_requires_grad || ctx->grad_requires_grad)) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    ctx->SaveTensorForBackward(inputs.at(0));
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarPowGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->resize(2);

    // z = x^a, dx = a * x^(a-1) * dz
    // grad_for_x  = out_grad * a * dz * [x^(a-1)]'
    // grad_for_dz = out_grad * [x^a]'

    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(1);
      const auto operand_sub_one = ctx->operand - Scalar(1);
      in_grads->at(0) = JUST(
          functional::sequence_function(functional::Mul)
              .then(std::bind(functional::ScalarPowGrad, x, std::placeholders::_1, operand_sub_one))
              .then([&ctx](const std::shared_ptr<Tensor>& input) {
                return functional::ScalarMul(ctx->operand, input);
              })
              .call(grad, out_grads.at(0)));
    }
    if (ctx->grad_requires_grad) {
      in_grads->at(1) = JUST(functional::ScalarPowGrad(x, out_grads.at(0), ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

class ScalarReversePowGradGrad : public OpExprGradFunction<ScalarPowGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarPowGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!(ctx->x_requires_grad || ctx->grad_requires_grad)) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }
    ctx->SaveTensorForBackward(inputs.at(0));
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(outputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarPowGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->resize(2);

    // z = a^x, dx = a^x * ln(a) * dz
    // grad_for_x  = out_grad * dz * a^x * ln(a) * ln(a)
    // grad_for_dz = out_grad * [a^x]'

    if (ctx->x_requires_grad) {
      const auto& dx = ctx->SavedTensors().at(1);
      const auto log_operand = std::log(ctx->operand.As<double>());
      in_grads->at(0) = JUST(functional::sequence_function(functional::Mul)
                                 .then([&log_operand](const std::shared_ptr<Tensor>& input) {
                                   return functional::ScalarMul(log_operand, input);
                                 })
                                 .call(dx, out_grads.at(0)));
    }
    if (ctx->grad_requires_grad) {
      in_grads->at(1) = JUST(functional::ScalarReversePowGrad(x, out_grads.at(0), ctx->operand));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_pow_grad", ScalarPowGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_reverse_pow_grad", ScalarReversePowGradGrad);

}  // namespace one
}  // namespace oneflow
