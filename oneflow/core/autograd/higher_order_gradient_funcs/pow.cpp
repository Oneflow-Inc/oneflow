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

#include <functional>
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct BroadcastPowXGradGradCaptureState : public AutoGradCaptureState {
  bool dz_requires_grad = false;
  bool x_requires_grad = false;
  bool y_requires_grad = false;
  bool z_requires_grad = false;

  size_t dz_index = 0;
  size_t x_index = 1;
  size_t y_index = 2;
  size_t z_index = 3;
};

class BroadcastPowXGradGrad : public OpExprGradFunction<BroadcastPowXGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(BroadcastPowXGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dz, x, y, z
    CHECK_EQ_OR_RETURN(inputs.size(), 4);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->dz_requires_grad = inputs.at(0)->requires_grad();
    ctx->x_requires_grad = inputs.at(1)->requires_grad();
    ctx->y_requires_grad = inputs.at(2)->requires_grad();
    ctx->z_requires_grad = inputs.at(3)->requires_grad();
    if (!(ctx->dz_requires_grad || ctx->x_requires_grad || ctx->y_requires_grad
          || ctx->z_requires_grad)) {
      return Maybe<void>::Ok();
    }

    ctx->dz_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(1));
    ctx->y_index = ctx->SaveTensorForBackward(inputs.at(2));
    ctx->z_index = ctx->SaveTensorForBackward(inputs.at(3));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const BroadcastPowXGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(4);
    const auto& dz = ctx->SavedTensors().at(ctx->dz_index);
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    const auto& y = ctx->SavedTensors().at(ctx->y_index);
    const auto& z = ctx->SavedTensors().at(ctx->z_index);

    // dx = y * x^(y-1) * dz = z / x * y * dz
    // grad_for_dz = out_grads * z / x * y
    // grad_for_x  = out_grads * -1 / (x*x) * z * y * dz
    // grad_for_y  = out_grads * z / x * dz
    // grad_for_z  = out_grads / x * y * dz

    if (ctx->dz_requires_grad || ctx->x_requires_grad) {
      const auto& tmp = JUST(functional::sequence_function(functional::Mul)
                                 .then(std::bind(functional::Mul, std::placeholders::_1, z))
                                 .call(out_grads.at(0), y));
      if (ctx->dz_requires_grad) {
        in_grads->at(0) = JUST(functional::sequence_function(functional::ReciprocalNoNan)
                                   .then(std::bind(functional::Mul, std::placeholders::_1, tmp))
                                   .call(x));
      }
      if (ctx->x_requires_grad) {
        in_grads->at(1) =
            JUST(functional::sequence_function(functional::Mul)
                     .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, x))
                     .then(std::bind(functional::ReciprocalNoNanGrad, x, std::placeholders::_1))
                     .call(tmp, dz));
      }
    }
    if (ctx->y_requires_grad || ctx->z_requires_grad) {
      const auto& tmp =
          JUST(functional::sequence_function(functional::ReciprocalNoNan)
                   .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                   .then(std::bind(functional::Mul, std::placeholders::_1, dz))
                   .call(x));
      if (ctx->y_requires_grad) {
        in_grads->at(2) =
            JUST(functional::sequence_function(functional::Mul)
                     .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, y))
                     .call(tmp, z));
      }
      if (ctx->z_requires_grad) { in_grads->at(3) = JUST(functional::Mul(tmp, y)); }
    }
    return Maybe<void>::Ok();
  }
};

struct BroadcastPowYGradGradCaptureState : public AutoGradCaptureState {
  bool dz_requires_grad = false;
  bool x_requires_grad = false;
  bool z_requires_grad = false;

  size_t dz_index = 0;
  size_t x_index = 1;
  size_t y_index = 2;
  size_t z_index = 3;
};

class BroadcastPowYGradGrad : public OpExprGradFunction<BroadcastPowYGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(BroadcastPowYGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dz, x, y, z  note: y is only used in shape/dtype infer
    CHECK_EQ_OR_RETURN(inputs.size(), 4);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->dz_requires_grad = inputs.at(0)->requires_grad();
    ctx->x_requires_grad = inputs.at(1)->requires_grad();
    ctx->z_requires_grad = inputs.at(3)->requires_grad();
    if (!(ctx->dz_requires_grad || ctx->x_requires_grad || ctx->z_requires_grad)) {
      return Maybe<void>::Ok();
    }

    ctx->dz_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(1));
    if (ctx->dz_requires_grad) { ctx->y_index = ctx->SaveTensorForBackward(inputs.at(2)); }
    ctx->z_index = ctx->SaveTensorForBackward(inputs.at(3));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const BroadcastPowYGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(4);
    const auto& dz = ctx->SavedTensors().at(ctx->dz_index);
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    const auto& z = ctx->SavedTensors().at(ctx->z_index);

    // dy = x^y * ln(x) * dz = z * ln(x) * dz
    // grad_for_dz = out_grads * ln(x) * z
    // grad_for_x  = out_grads * z * dz / x
    // grad_for_z  = out_grads * ln(x) * dz

    if (ctx->dz_requires_grad || ctx->z_requires_grad) {
      const auto& tmp =
          JUST(functional::sequence_function(functional::Log)
                   .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                   .call(x));
      if (ctx->dz_requires_grad) { in_grads->at(0) = JUST(functional::Mul(tmp, z)); }
      if (ctx->z_requires_grad) { in_grads->at(3) = JUST(functional::Mul(tmp, dz)); }
    }
    if (ctx->x_requires_grad) {
      in_grads->at(1) =
          JUST(functional::sequence_function(functional::Mul)
                   .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                   .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, x))
                   .then(std::bind(functional::Div, std::placeholders::_1, x))
                   .call(z, dz));
    }
    return Maybe<void>::Ok();
  }
};

struct PowXGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool y_requires_grad = false;
  bool dz_requires_grad = false;

  size_t x_index = 0;
  size_t y_index = 1;
  size_t dz_index = 2;
};

class PowXGradGrad : public OpExprGradFunction<PowXGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(PowXGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, y, dz
    CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    ctx->dz_requires_grad = inputs.at(2)->requires_grad();
    if (!(ctx->dz_requires_grad || ctx->x_requires_grad || ctx->y_requires_grad)) {
      return Maybe<void>::Ok();
    }

    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      ctx->dz_index = ctx->SaveTensorForBackward(inputs.at(2));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PowXGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(3);
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    const auto& y = ctx->SavedTensors().at(ctx->y_index);

    // dx = y * x^(y-1) * dz
    // grad_for_x  = out_grads * dz * y * [x^(y-1)]'
    // grad_for_y  = out_grads * dz * [x^(y-1) * (1 + y * ln(x))]
    // grad_for_dz = out_grads * y * x^(y-1)

    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      const auto& dz = ctx->SavedTensors().at(ctx->dz_index);
      const auto& y_sub_one = JUST(functional::ScalarSub(y, 1, /*alpha=*/1, /*inplace=*/false));
      if (ctx->x_requires_grad) {
        in_grads->at(0) = JUST(functional::sequence_function(functional::PowXGrad)
                                   .then(std::bind(functional::Mul, std::placeholders::_1, y))
                                   .then(std::bind(functional::Mul, std::placeholders::_1, dz))
                                   .call(x, y_sub_one, out_grads.at(0)));
      }
      if (ctx->y_requires_grad) {
        in_grads->at(1) =
            JUST(functional::sequence_function(functional::Log)
                     .then(std::bind(functional::Mul, std::placeholders::_1, y))
                     .then([](const std::shared_ptr<Tensor>& input) {
                       return functional::ScalarAdd(1, input, /*alpha=*/1);
                     })
                     .then(std::bind(functional::Mul, std::placeholders::_1,
                                     JUST(functional::Pow(x, y_sub_one))))
                     .then(std::bind(functional::Mul, std::placeholders::_1, dz))
                     .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                     .call(x));
      }
    }
    if (ctx->dz_requires_grad) {
      in_grads->at(2) = JUST(functional::PowXGrad(x, y, out_grads.at(0)));
    }
    return Maybe<void>::Ok();
  }
};

struct PowYGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool y_requires_grad = false;
  bool dz_requires_grad = false;

  size_t x_index = 0;
  size_t y_index = 1;
  size_t dz_index = 2;
  size_t dy_index = 3;
};

class PowYGradGrad : public OpExprGradFunction<PowYGradGradCaptureState> {
 public:
  // dy = x^y*ln(x)*dz = z*ln(x)*dz
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(PowYGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // x, y, dz
    CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    ctx->dz_requires_grad = inputs.at(2)->requires_grad();

    if (!(ctx->dz_requires_grad || ctx->x_requires_grad || ctx->y_requires_grad)) {
      return Maybe<void>::Ok();
    }
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
    }
    if (ctx->x_requires_grad) { ctx->dz_index = ctx->SaveTensorForBackward(inputs.at(2)); }
    if (ctx->y_requires_grad) { ctx->dy_index = ctx->SaveTensorForBackward(outputs.at(0)); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PowYGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(3);
    const auto& x = ctx->SavedTensors().at(ctx->x_index);

    // dy = x^y * ln(x) * dz = z * ln(x) * dz
    // grad_for_x  = out_grads * dz * [x^(y-1) * (1 + y * ln(x))]
    // grad_for_y  = out_grads * dy' = out_grads * dy * ln(x)
    // grad_for_dz = out_grads * x^y * ln(x)

    if (ctx->x_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      const auto& dz = ctx->SavedTensors().at(ctx->dz_index);
      const auto& y_sub_one = JUST(functional::ScalarSub(y, 1, /*alpha=*/1, /*inplace=*/false));
      in_grads->at(0) =
          JUST(functional::sequence_function(functional::Log)
                   .then(std::bind(functional::Mul, std::placeholders::_1, y))
                   .then([](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarAdd(1, input, /*alpha=*/1);
                   })
                   .then(std::bind(functional::Mul, std::placeholders::_1,
                                   JUST(functional::Pow(x, y_sub_one))))
                   .then(std::bind(functional::Mul, std::placeholders::_1, dz))
                   .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                   .call(x));
    }

    if (ctx->y_requires_grad) {
      const auto& dy = ctx->SavedTensors().at(ctx->dy_index);
      in_grads->at(1) =
          JUST(functional::sequence_function(functional::Log)
                   .then(std::bind(functional::Mul, std::placeholders::_1, dy))
                   .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
                   .call(x));
    }

    if (ctx->dz_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      in_grads->at(2) = JUST(functional::PowYGrad(x, y, out_grads.at(0)));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("pow_x_grad", PowXGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("pow_y_grad", PowYGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_pow_x_grad", BroadcastPowXGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_pow_y_grad", BroadcastPowYGradGrad);

}  // namespace one
}  // namespace oneflow
