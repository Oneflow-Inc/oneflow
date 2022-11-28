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

struct DivGradGradCaptureState : public AutoGradCaptureState {
  bool y_requires_grad = false;
  bool z_requires_grad = false;
  bool grad_requires_grad = false;

  size_t y_index = 0;
  size_t z_index = 1;
  size_t grad_index = 2;
};

class DivGradGrad : public OpExprGradFunction<DivGradGradCaptureState> {
  // div_grad    = -x/(y*y)*dz = -z/y*dz
  // div_grad_y  = out_grad * z*dz/(y*y)
  // div_grad_z  = out_grad * -dz/y
  // div_grad_dz = out_grad * -z/y
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(DivGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dz, z, y
    CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->grad_requires_grad = inputs.at(0)->requires_grad();
    ctx->z_requires_grad = inputs.at(1)->requires_grad();
    ctx->y_requires_grad = inputs.at(2)->requires_grad();

    ctx->y_index = ctx->SaveTensorForBackward(inputs.at(2));
    if (ctx->y_requires_grad || ctx->grad_requires_grad) {
      ctx->z_index = ctx->SaveTensorForBackward(inputs.at(1));
    }
    if (ctx->y_requires_grad || ctx->z_requires_grad) {
      ctx->grad_index = ctx->SaveTensorForBackward(inputs.at(0));
    }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const DivGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(3);
    const auto& y = ctx->SavedTensors().at(ctx->y_index);

    if (ctx->grad_requires_grad) {
      const auto& z = ctx->SavedTensors().at(ctx->z_index);
      in_grads->at(0) = JUST(functional::sequence_function(functional::Mul)
                                 .then(functional::Negative)
                                 .then(std::bind(functional::Div, std::placeholders::_1, y))
                                 .call(out_grads.at(0), z));
    }
    if (ctx->z_requires_grad) {
      const auto& grad = ctx->SavedTensors().at(ctx->grad_index);
      in_grads->at(1) = JUST(functional::sequence_function(functional::Mul)
                                 .then(functional::Negative)
                                 .then(std::bind(functional::Div, std::placeholders::_1, y))
                                 .call(out_grads.at(0), grad));
    }
    if (ctx->y_requires_grad) {
      const auto& z = ctx->SavedTensors().at(ctx->z_index);
      const auto& grad = ctx->SavedTensors().at(ctx->grad_index);
      in_grads->at(2) = JUST(
          functional::sequence_function(functional::Mul)
              .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, y))
              .then(std::bind(functional::Mul, std::placeholders::_1, out_grads.at(0)))
              .then(std::bind(functional::Div, std::placeholders::_1, JUST(functional::Square(y))))
              .call(z, grad));
    }

    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_div_grad", DivGradGrad);

}  // namespace one
}  // namespace oneflow
