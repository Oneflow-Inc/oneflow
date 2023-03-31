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
#include <cstddef>
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct BaseActivationGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
};

typedef Maybe<one::Tensor> (*NoParamActivationBwFunc)(const std::shared_ptr<one::Tensor>&,
                                                      const std::shared_ptr<one::Tensor>&);

template<NoParamActivationBwFunc BwFunc, NoParamActivationBwFunc BwBwFunc>
class NoParamActivationGradGrad : public OpExprGradFunction<BaseActivationGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(BaseActivationGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, x
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(1)->requires_grad();
    ctx->grad_requires_grad = inputs.at(0)->requires_grad();

    if (!ctx->x_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ctx->SaveTensorForBackward(inputs.at(1));
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const BaseActivationGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors().at(0);

    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(1);
      in_grads->at(1) = JUST(functional::Mul(out_grads.at(0), JUST(BwBwFunc(x, grad))));
    }
    if (ctx->grad_requires_grad) { in_grads->at(0) = JUST(BwFunc(out_grads.at(0), x)); }
    return Maybe<void>::Ok();
  }
};

#define INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS(op_type_name, op_cls)                     \
  class op_cls##GradGradCls final                                                                  \
      : public NoParamActivationGradGrad<functional::op_cls##Grad, functional::op_cls##GradGrad> { \
  };                                                                                               \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, op_cls##GradGradCls);

// first order backward param: (dy, x)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("mish_grad", Mish)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("gelu_grad", Gelu)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("silu_grad", Silu)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("selu_grad", Selu)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("softsign_grad", SoftSign)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("hardsigmoid_grad", HardSigmoid)
INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS("hardswish_grad", HardSwish)

#undef INSTANTIAT_AND_REGISTER_NOPARAM_ACTIVATION_CLASS

struct HardShrinkGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool grad_requires_grad = false;
  double lambd = 0.5;
};

class HardShrinkGradGrad : public OpExprGradFunction<HardShrinkGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(HardShrinkGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // y, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->y_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->y_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->lambd = JUST(composed_attrs.GetAttr<double>("lambd"));
    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HardShrinkGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (ctx->y_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(1) = JUST(functional::HardShrinkGrad(y, out_grads.at(0), ctx->lambd));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct SoftShrinkGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool grad_requires_grad = false;
  double alpha = 0.5;
};

class SoftShrinkGradGrad : public OpExprGradFunction<SoftShrinkGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(SoftShrinkGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // y, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->y_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->y_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SoftShrinkGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (ctx->y_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(1) = JUST(functional::SoftShrinkGrad(y, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct ReluGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool grad_requires_grad = false;
};

class ReluGradGrad : public OpExprGradFunction<ReluGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(ReluGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, y
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    ctx->grad_requires_grad = inputs.at(0)->requires_grad();

    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const ReluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->y_requires_grad) { in_grads->at(1) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::ReluGrad(out_grads.at(0), y));
    }
    return Maybe<void>::Ok();
  }
};

struct LeakyReluGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  float alpha = 0.01;
};

class LeakyReluGradGrad : public OpExprGradFunction<LeakyReluGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(LeakyReluGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->x_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<float>("alpha"));

    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LeakyReluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(1) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct SoftplusGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  double beta = 1.0;
  double threshold = 20.0;
};

class SoftplusGradGrad : public OpExprGradFunction<SoftplusGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SoftplusGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->x_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->beta = JUST(composed_attrs.GetAttr<double>("beta"));
    ctx->threshold = JUST(composed_attrs.GetAttr<double>("threshold"));

    ctx->SaveTensorForBackward(inputs.at(0));
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SoftplusGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors().at(0);

    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(1);
      in_grads->at(0) = JUST(functional::Mul(
          out_grads.at(0), JUST(functional::SoftplusGradGrad(x, grad, ctx->beta, ctx->threshold))));
    }
    if (ctx->grad_requires_grad) {
      in_grads->at(1) =
          JUST(functional::SoftplusGrad(x, out_grads.at(0), ctx->beta, ctx->threshold));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct HardTanhGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool grad_requires_grad = false;
  double min_val = -1.0;
  double max_val = 1.0;
};

class HardTanhGradGrad : public OpExprGradFunction<HardTanhGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(HardTanhGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // y, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->y_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->y_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->min_val = JUST(composed_attrs.GetAttr<double>("min_val"));
    ctx->max_val = JUST(composed_attrs.GetAttr<double>("max_val"));
    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HardTanhGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (ctx->y_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(1) =
          JUST(functional::HardTanhGrad(y, out_grads.at(0), ctx->min_val, ctx->max_val));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct EluGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  double alpha = 1.0;
};

class EluGradGrad : public OpExprGradFunction<EluGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(EluGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));

    if (!ctx->x_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }
    ctx->SaveTensorForBackward(inputs.at(0));
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const EluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors().at(0);

    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(1);
      in_grads->at(0) = JUST(
          functional::Mul(out_grads.at(0), JUST(functional::EluGradGrad(x, grad, ctx->alpha))));
    }
    if (ctx->grad_requires_grad) {
      in_grads->at(1) = JUST(functional::EluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

class CeluGradGrad : public EluGradGrad {
 public:
  Maybe<void> Apply(const EluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& y = ctx->SavedTensors().at(0);

    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(1);
      in_grads->at(0) = JUST(
          functional::CeluGradGrad(y, JUST(functional::Mul(out_grads.at(0), (grad))), ctx->alpha));
    }
    if (ctx->grad_requires_grad) {
      in_grads->at(1) = JUST(functional::CeluGrad(y, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }
};

struct PReluGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
  bool alpha_requires_grad = false;
  size_t grad_index = 0;
  size_t input_index = 1;
  size_t alpha_index = 2;
};

class PReluGradGrad : public OpExprGradFunction<PReluGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(PReluGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, x, alpha
    CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)

    ctx->grad_requires_grad = inputs.at(0)->requires_grad();   // grad
    ctx->input_requires_grad = inputs.at(1)->requires_grad();  // input
    ctx->alpha_requires_grad = inputs.at(2)->requires_grad();  // alpha

    ctx->input_index = ctx->SaveTensorForBackward(inputs.at(1));
    ctx->alpha_index = ctx->SaveTensorForBackward(inputs.at(2));
    ctx->grad_index = ctx->SaveTensorForBackward(inputs.at(0));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PReluGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(3);

    const auto& input = ctx->SavedTensors().at(ctx->input_index);
    const auto& alpha = ctx->SavedTensors().at(ctx->alpha_index);
    const auto& grad = ctx->SavedTensors().at(ctx->grad_index);
    const auto& grad_for_input = out_grads.at(0);
    const auto& grad_for_alpha = out_grads.at(1);
    const auto& condition = JUST(functional::ScalarLogicalLess(input, Scalar(0.0)));
    const auto& zero_grad = JUST(functional::ZerosLike(alpha));  // alpha can broadcast to input

    if (ctx->grad_requires_grad) {
      auto input_mul_grad = JUST(functional::Mul(alpha, grad_for_input));
      auto alpha_mul_grad = JUST(functional::Mul(input, grad_for_alpha));
      auto result = JUST(functional::Add(input_mul_grad, alpha_mul_grad, /*alpha=*/Scalar(1.0),
                                         /*inplace*/ false));
      in_grads->at(0) = JUST(functional::Where(condition, result, grad_for_input));
    }
    if (ctx->input_requires_grad) {
      auto result = JUST(functional::Mul(grad, grad_for_alpha));
      in_grads->at(1) = JUST(functional::Where(condition, result, zero_grad));
    }
    if (ctx->alpha_requires_grad) {
      auto result = JUST(functional::Mul(grad, grad_for_input));
      in_grads->at(2) = JUST(functional::Where(condition, result, zero_grad));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

struct ThresholdGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
  double threshold = 0.0;
};

class ThresholdGradGrad : public OpExprGradFunction<ThresholdGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ThresholdGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, dy
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->grad_requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->x_requires_grad && !ctx->grad_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->threshold = JUST(composed_attrs.GetAttr<double>("threshold_val"));

    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ThresholdGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::ZerosLike(out_grads.at(0))); }
    if (ctx->grad_requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(1) = JUST(functional::ThresholdGrad(x, out_grads.at(0), ctx->threshold));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("relu_grad", ReluGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("elu_grad", EluGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("celu_grad", CeluGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("prelu_grad", PReluGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("hardshrink_grad", HardShrinkGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("softshrink_grad", SoftShrinkGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("leaky_relu_grad", LeakyReluGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("hardtanh_grad", HardTanhGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("threshold_grad", ThresholdGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("softplus_grad", SoftplusGradGrad);

}  // namespace one
}  // namespace oneflow
