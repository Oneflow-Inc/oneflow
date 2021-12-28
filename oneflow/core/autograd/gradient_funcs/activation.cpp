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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BaseActivationCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class BaseActivation : public OpExprGradFunction<BaseActivationCaptureState> {
 public:
  Maybe<void> Capture(BaseActivationCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (state->requires_grad) { state->SaveTensorForBackward(inputs.at(0)); }
    return Maybe<void>::Ok();
  }
};

class Silu : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SiluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Mish : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::MishGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Selu : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SeluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class Softsign : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::SoftSignGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class GeLU : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::GeluGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class HardSigmoid : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::HardSigmoidGrad(out_grads.at(0), x));
    }
    return Maybe<void>::Ok();
  }
};

class HardSwish : public BaseActivation {
 public:
  Maybe<void> Apply(const BaseActivationCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
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
  Maybe<void> Capture(ReLUCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (state->requires_grad) { state->SaveTensorForBackward(outputs.at(0)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReLUCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& y = state->SavedTensors().at(0);
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
  Maybe<void> Capture(LeakyReluCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<LeakyReluOp>());
    state->alpha = op_ctx->alpha();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LeakyReluCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), state->alpha));
    }
    return Maybe<void>::Ok();
  }
};

struct HardTanhCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  double min_val;
  double max_val;
};

class HardTanh : public OpExprGradFunction<HardTanhCaptureState> {
 public:
  Maybe<void> Capture(HardTanhCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<HardtanhOp>());
    state->min_val = op_ctx->min_val();
    state->max_val = op_ctx->max_val();
    state->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HardTanhCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& y = state->SavedTensors().at(0);
      in_grads->at(0) =
          JUST(functional::HardTanhGrad(y, out_grads.at(0), state->min_val, state->max_val));
    }
    return Maybe<void>::Ok();
  }
};

struct EluCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  double alpha;
};

class Elu : public OpExprGradFunction<EluCaptureState> {
 public:
  Maybe<void> Capture(EluCaptureState* state, const TensorTuple& inputs, const TensorTuple& outputs,
                      const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<EluOp>());
    state->alpha = op_ctx->alpha();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const EluCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::EluGrad(x, out_grads.at(0), state->alpha));
    }
    return Maybe<void>::Ok();
  }
};

struct CeluCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  double alpha = 1.0;
};

class Celu : public OpExprGradFunction<CeluCaptureState> {
 public:
  Maybe<void> Capture(CeluCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = JUST(ctx->dyn_cast<CeluOp>());
    state->alpha = op_ctx->alpha();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CeluCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      const auto& x = state->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::CeluGrad(x, out_grads.at(0), state->alpha));
    }
    return Maybe<void>::Ok();
  }
};

struct PReLUCaptureState : public AutoGradCaptureState {
  bool input_requires_grad;
  bool alpha_requires_grad;
};

class PReLU : public OpExprGradFunction<PReLUCaptureState> {
 public:
  Maybe<void> Capture(PReLUCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->input_requires_grad = inputs.at(0)->requires_grad();  // input
    state->alpha_requires_grad = inputs.at(1)->requires_grad();  // alpha
    state->SaveTensorForBackward(inputs.at(0));
    state->SaveTensorForBackward(inputs.at(1));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PReLUCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& dy = out_grads.at(0);
    const auto& x = state->SavedTensors().at(0);
    const auto& alpha = state->SavedTensors().at(1);
    in_grads->resize(2);
    if (state->input_requires_grad || state->alpha_requires_grad) {
      const auto& grads = JUST(functional::PReluGrad(dy, x, alpha));
      if (state->input_requires_grad) { in_grads->at(0) = grads->at(0); }
      if (state->alpha_requires_grad) { in_grads->at(1) = grads->at(1); }
    }
    return Maybe<void>::Ok();
  }
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

}  // namespace one
}  // namespace oneflow
