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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct WhereCaptureState : public AutoGradCaptureState {
  bool requires_grad_x;
  bool requires_grad_y;
};

struct WhereScalarCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class Where : public OpExprGradFunction<WhereCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(WhereCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const WhereCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Where::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> Where::Capture(WhereCaptureState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_x = inputs.at(1)->requires_grad();
  ctx->requires_grad_y = inputs.at(2)->requires_grad();
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));  // condition
  ctx->SaveTensorForBackward(inputs.at(1));  // x
  ctx->SaveTensorForBackward(inputs.at(2));  // y
  return Maybe<void>::Ok();
}

Maybe<void> Where::Apply(const WhereCaptureState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<oneflow::one::Tensor>& condition = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(2);

  std::shared_ptr<oneflow::one::Tensor> zero_out = JUST(functional::ZerosLike(x));
  in_grads->resize(3);
  if (ctx->requires_grad_x) {
    auto broad_x_grad = JUST(functional::Where(condition, out_grads.at(0), zero_out));
    in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(broad_x_grad, x));
  }
  if (ctx->requires_grad_y) {
    auto broad_y_grad = JUST(functional::Where(condition, zero_out, out_grads.at(0)));
    in_grads->at(2) = JUST(functional::BroadcastReduceSumLike(broad_y_grad, y));
  }
  return Maybe<void>::Ok();
}

class WhereScalar : public OpExprGradFunction<WhereScalarCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(WhereScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->requires_grad = inputs.at(1)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }
};

class WhereScalarX : public WhereScalar {
 public:
  Maybe<void> Apply(const WhereScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& condition = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(1);

    std::shared_ptr<oneflow::one::Tensor> zero_out = JUST(functional::ZerosLike(y));
    in_grads->resize(2);
    auto broad_y_grad = JUST(functional::Where(condition, zero_out, out_grads.at(0)));
    in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(broad_y_grad, y));
    return Maybe<void>::Ok();
  }
};

class WhereScalarY : public WhereScalar {
 public:
  Maybe<void> Apply(const WhereScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& condition = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(1);

    std::shared_ptr<oneflow::one::Tensor> zero_out = JUST(functional::ZerosLike(x));
    in_grads->resize(2);
    auto broad_x_grad = JUST(functional::Where(condition, out_grads.at(0), zero_out));
    in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(broad_x_grad, x));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("where", Where);
REGISTER_OP_EXPR_GRAD_FUNCTION("where_scalar_x", WhereScalarX);
REGISTER_OP_EXPR_GRAD_FUNCTION("where_scalar_y", WhereScalarY);

}  // namespace one
}  // namespace oneflow
