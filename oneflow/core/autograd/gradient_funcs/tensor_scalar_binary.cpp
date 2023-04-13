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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct TensorScalarCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
  bool scalar_requires_grad;
};

class TensorScalarAddOrSub : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  TensorScalarAddOrSub() = default;
  virtual ~TensorScalarAddOrSub() = default;

  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
};

Maybe<void> TensorScalarAddOrSub::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarAddOrSub::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                          const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  return Maybe<void>::Ok();
}

class TensorScalarAdd : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::Identity(out_grads.at(0))); }
    if (ctx->scalar_requires_grad) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      in_grads->at(1) = JUST(functional::ReduceSum(out_grads.at(0), axes_vec, false));
    }
    return Maybe<void>::Ok();
  }
};

class TensorScalarSub : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(functional::Identity(out_grads.at(0))); }
    if (ctx->scalar_requires_grad) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      const auto& reduce_sum =
          JUST(functional::ReduceSum(out_grads.at(0), axes_vec, /*keepdims=*/false));
      in_grads->at(1) = JUST(functional::ScalarMul(reduce_sum, /*other=*/1.0, false));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_add_by_tensor", TensorScalarAdd);
REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_sub_by_tensor", TensorScalarSub);

class TensorScalarMul : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> TensorScalarMul::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
  if (ctx->scalar_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Apply(const TensorScalarCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (ctx->x_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::Mul(out_grads.at(0), scalar));
  }
  if (ctx->scalar_requires_grad) {
    const auto& x = ctx->SavedTensors().at(ctx->x_requires_grad);
    const auto& y = JUST(functional::Mul(out_grads.at(0), x));
    int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
    std::vector<int32_t> axes_vec(num_axes);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
    in_grads->at(1) = JUST(functional::ReduceSum(y, axes_vec, /*keepdims=*/false));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_mul_by_tensor", TensorScalarMul);

class TensorScalarDiv : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> tensor_scalar_div_op_;
  std::shared_ptr<OpExpr> broadcast_div_grad_op_;
};

Maybe<void> TensorScalarDiv::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  if (ctx->x_requires_grad || ctx->scalar_requires_grad) {
    ctx->SaveTensorForBackward(inputs.at(1));
  }
  if (ctx->scalar_requires_grad) { ctx->SaveTensorForBackward(outputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Apply(const TensorScalarCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (ctx->x_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::Div(out_grads.at(0), scalar));
  }
  if (ctx->scalar_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->at(1) = JUST(functional::DivGrad(out_grads.at(0), y, scalar));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_div_by_tensor", TensorScalarDiv);

}  // namespace one
}  // namespace oneflow
