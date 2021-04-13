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
#include "oneflow/core/autograd/gradient_funcs/utility.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_dispatch.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

class TensorScalarAddOrSub : public OpExprGradFunction {
 public:
  TensorScalarAddOrSub() = default;
  virtual ~TensorScalarAddOrSub() = default;

  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;

 protected:
  std::string op_name_;
  std::shared_ptr<OpExpr> identity_op_;
  mutable bool x_requires_grad_;
  mutable bool scalar_requires_grad_;
};

Maybe<void> TensorScalarAddOrSub::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  identity_op_ = JUST(op_expr_helper::IdentityOp(GradientOpName(op_name_ + "_x")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarAddOrSub::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                          const TensorTuple& outputs) const {
  x_requires_grad_ = inputs.at(0)->requires_grad();
  scalar_requires_grad_ = inputs.at(1)->requires_grad();
  return Maybe<void>::Ok();
}

class TensorScalarAdd : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (x_requires_grad_) {
      in_grads->at(0) = JUST(Dispatch<Tensor>(*identity_op_, {out_grads.at(0)}));
    }
    if (scalar_requires_grad_) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      const auto& reduce_sum_op = JUST(op_expr_helper::ReduceSumOp(
          axes_vec, /*keepdims=*/false, GradientOpName(op_name_ + "_scalar")));
      in_grads->at(1) = JUST(Dispatch<Tensor>(*reduce_sum_op, {out_grads.at(0)}));
    }
    return Maybe<void>::Ok();
  }
};

class TensorScalarSub : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (x_requires_grad_) {
      in_grads->at(0) = JUST(Dispatch<Tensor>(*identity_op_, {out_grads.at(0)}));
    }
    if (scalar_requires_grad_) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      const auto& reduce_sum_op = JUST(op_expr_helper::ReduceSumOp(
          axes_vec, /*keepdims=*/false, GradientOpName(op_name_ + "_scalar_reduce_sum")));
      const auto& reduce_sum = JUST(Dispatch<Tensor>(*reduce_sum_op, {out_grads.at(0)}));
      const auto& scalar_mul_op =
          JUST(op_expr_helper::ScalarMulOp<float>(-1.f, GradientOpName(op_name_ + "_scalar")));
      in_grads->at(1) = JUST(Dispatch<Tensor>(*scalar_mul_op, {reduce_sum}));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_add_by_tensor", TensorScalarAdd);
REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_sub_by_tensor", TensorScalarSub);

class TensorScalarMul : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  std::shared_ptr<OpExpr> tensor_scalar_mul_op_;
  std::shared_ptr<OpExpr> multiply_op_;
  mutable bool x_requires_grad_;
  mutable bool scalar_requires_grad_;
};

Maybe<void> TensorScalarMul::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  tensor_scalar_mul_op_ =
      JUST(op_expr_helper::ScalarMulByTensorOp(GradientOpName(op_name_ + "_x")));
  multiply_op_ = JUST(op_expr_helper::MultiplyOp(GradientOpName(op_name_ + "_scalar_mul")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs) const {
  x_requires_grad_ = inputs.at(0)->requires_grad();
  scalar_requires_grad_ = inputs.at(1)->requires_grad();
  if (x_requires_grad_) { ctx->SaveTensorForBackward(inputs.at(1)); }
  if (scalar_requires_grad_) { ctx->SaveTensorForBackward(inputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (x_requires_grad_) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(Dispatch<Tensor>(*tensor_scalar_mul_op_, {out_grads.at(0), scalar}));
  }
  if (scalar_requires_grad_) {
    const auto& x = ctx->SavedTensors().at(x_requires_grad_);
    const auto& y = JUST(Dispatch<Tensor>(*multiply_op_, {out_grads.at(0), x}));
    int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
    std::vector<int32_t> axes_vec(num_axes);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
    const auto& reduce_sum_op = JUST(op_expr_helper::ReduceSumOp(
        axes_vec, /*keepdims=*/false, GradientOpName(op_name_ + "_scalar_reduce_sum")));
    in_grads->at(1) = JUST(Dispatch<Tensor>(*reduce_sum_op, {y}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_mul_by_tensor", TensorScalarMul);

class TensorScalarDiv : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  std::shared_ptr<OpExpr> tensor_scalar_div_op_;
  std::shared_ptr<OpExpr> broadcast_div_op_;
  mutable bool x_requires_grad_;
  mutable bool scalar_requires_grad_;
};

Maybe<void> TensorScalarDiv::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  tensor_scalar_div_op_ =
      JUST(op_expr_helper::ScalarDivByTensorOp(GradientOpName(op_name_ + "_x")));
  broadcast_div_op_ = JUST(op_expr_helper::BroadcastDivOp(GradientOpName(op_name_ + "_scalar")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs) const {
  x_requires_grad_ = inputs.at(0)->requires_grad();
  scalar_requires_grad_ = inputs.at(1)->requires_grad();
  if (x_requires_grad_ || scalar_requires_grad_) { ctx->SaveTensorForBackward(inputs.at(1)); }
  if (scalar_requires_grad_) { ctx->SaveTensorForBackward(outputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (x_requires_grad_) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(Dispatch<Tensor>(*tensor_scalar_div_op_, {out_grads.at(0), scalar}));
  }
  if (scalar_requires_grad_) {
    const auto& scalar = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->at(1) = JUST(Dispatch<Tensor>(*broadcast_div_op_, {out_grads.at(0), y, scalar}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_div_by_tensor", TensorScalarDiv);

}  // namespace one
}  // namespace oneflow
