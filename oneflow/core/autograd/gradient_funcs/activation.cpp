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

struct BaseActivationCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class BaseActivation : public OpExprGradFunction<BaseActivationCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(BaseActivationCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (ctx->requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }
};

class Silu : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SiluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Mish : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::MishGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Selu : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SeluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Softsign : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SoftSignGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class GeLU : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::GeluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class HardSigmoid : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::HardSigmoidGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class HardSwish : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::HardSwishGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

// ===== Activation with parms ====
struct ReLUCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class ReLU : public OpExprGradFunction<ReLUCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(ReLUCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (ctx->requires_grad) { ctx->SaveTensorForBackward(outputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReLUCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& y = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::ReluGrad(out_grads.at(0), y));
    }
    return Maybe<void>::Ok();
  }
};

// ===== Activation with parms ====
struct LeakyReluCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float alpha;
};

class LeakyRelu : public OpExprGradFunction<LeakyReluCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(LeakyReluCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<float>("alpha"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LeakyReluCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct HardTanhCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  double min_val;
  double max_val;
};

class HardTanh : public OpExprGradFunction<HardTanhCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(HardTanhCaptureState* ctx, const TensorTuple& inputs,
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

  Maybe<void> Apply(const HardTanhCaptureState* ctx, const TensorTuple& out_grads,
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

struct EluCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  double alpha;
};

class Elu : public OpExprGradFunction<EluCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(EluCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const EluCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::EluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct CeluCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  double alpha = 1.0;
};

class Celu : public OpExprGradFunction<CeluCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CeluCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CeluCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::CeluGrad(x, out_grads.at(0), ctx->alpha));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct PReLUCaptureState : public AutoGradCaptureState {
  bool input_requires_grad;
  bool alpha_requires_grad;
};

class PReLU : public OpExprGradFunction<PReLUCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(PReLUCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();  // input
    ctx->alpha_requires_grad = inputs.at(1)->requires_grad();  // alpha
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PReLUCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& dy = out_grads.at(0);
    const auto& x = ctx->SavedTensors().at(0);
    const auto& alpha = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (ctx->input_requires_grad || ctx->alpha_requires_grad) {
      const auto& grads = JUST(functional::PReluGrad(dy, x, alpha));
      if (ctx->input_requires_grad) { in_grads->at(0) = grads->at(0); }
      if (ctx->alpha_requires_grad) { in_grads->at(1) = grads->at(1); }
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

struct TfPReLUCaptureState : public AutoGradCaptureState {
  bool input_requires_grad;
  bool alpha_requires_grad;
};

class TfPReLU : public OpExprGradFunction<TfPReLUCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(TfPReLUCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();  // input
    ctx->alpha_requires_grad = inputs.at(1)->requires_grad();  // alpha
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const TfPReLUCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& dy = out_grads.at(0);
    const auto& x = ctx->SavedTensors().at(0);
    const auto& alpha = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (ctx->input_requires_grad || ctx->alpha_requires_grad) {
      const auto& grads = JUST(functional::TfPReluGrad(dy, x, alpha));
      if (ctx->input_requires_grad) { in_grads->at(0) = grads->at(0); }
      if (ctx->alpha_requires_grad) { in_grads->at(1) = grads->at(1); }
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("silu", Silu);
REGISTER_OP_EXPR_GRAD_FUNCTION("mish", Mish);
REGISTER_OP_EXPR_GRAD_FUNCTION("selu", Selu);
REGISTER_OP_EXPR_GRAD_FUNCTION("softsign", Softsign);
REGISTER_OP_EXPR_GRAD_FUNCTION("relu", ReLU);
REGISTER_OP_EXPR_GRAD_FUNCTION("gelu", GeLU);
REGISTER_OP_EXPR_GRAD_FUNCTION("hardsigmoid", HardSigmoid);
REGISTER_OP_EXPR_GRAD_FUNCTION("hardswish", HardSwish);
REGISTER_OP_EXPR_GRAD_FUNCTION("leaky_relu", LeakyRelu);
REGISTER_OP_EXPR_GRAD_FUNCTION("hardtanh", HardTanh);
REGISTER_OP_EXPR_GRAD_FUNCTION("elu", Elu);
REGISTER_OP_EXPR_GRAD_FUNCTION("celu", Celu);
REGISTER_OP_EXPR_GRAD_FUNCTION("prelu", PReLU);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_prelu", TfPReLU);

}  // namespace one
}  // namespace oneflow
