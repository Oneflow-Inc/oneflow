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
#include "oneflow/core/framework/op_interpreter_util.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Tensor> ReduceSumLike(const std::shared_ptr<Tensor>& input,
                            const std::shared_ptr<Tensor>& like, const std::string& op_name) {
  const auto& in_shape = *(input->shape());
  const auto& like_shape = *(like->shape());
  TensorTuple inputs{input};
  std::shared_ptr<OpExpr> op(nullptr);
  if (in_shape == like_shape) {
    op = JUST(op_expr_helper::IdentityOp(op_name));
  } else {
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
    if (in_shape == left_extended_shape) {
      op = JUST(op_expr_helper::ReshapeLikeOp(op_name));
    } else {
      const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
      op = JUST(op_expr_helper::ReduceSumLikeOp(
          std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()}, op_name));
    }
    inputs.push_back(like);
  }
  return JUST(Dispatch<Tensor>(*op, inputs));
}

}  // namespace

class BroadcastBinary : public OpExprGradFunction {
 public:
  BroadcastBinary() = default;
  virtual ~BroadcastBinary() = default;

  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    op_name_ = fw_op_expr->op_name();
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }

 protected:
  std::string op_name_;
};

class BroadcastAdd : public BroadcastBinary {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      in_grads->at(0) = JUST(ReduceSumLike(out_grads.at(0), x, GradientOpName(op_name_ + "_x")));
    }
    if (y->requires_grad()) {
      in_grads->at(1) = JUST(ReduceSumLike(out_grads.at(0), y, GradientOpName(op_name_ + "_y")));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_add", BroadcastAdd);

class BroadcastSub : public BroadcastBinary {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      in_grads->at(0) = JUST(ReduceSumLike(out_grads.at(0), x, GradientOpName(op_name_ + "_x")));
    }
    if (y->requires_grad()) {
      const auto& scalar_mul_op =
          JUST(op_expr_helper::ScalarMulOp(-1.f, GradientOpName(op_name_ + "_y_scalar_mul")));
      const auto& scalar_mul = JUST(Dispatch<Tensor>(*scalar_mul_op, {out_grads.at(0)}));
      in_grads->at(1) = JUST(ReduceSumLike(scalar_mul, y, GradientOpName(op_name_ + "_y")));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_sub", BroadcastSub);

class BroadcastMul : public BroadcastBinary {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& broadcast_mul_op =
          JUST(op_expr_helper::BroadcastMulOp(GradientOpName(op_name_ + "x_broadcast_mul")));
      const auto& broadcast_mul = JUST(Dispatch<Tensor>(*broadcast_mul_op, {out_grads.at(0), y}));
      in_grads->at(0) = JUST(ReduceSumLike(broadcast_mul, x, GradientOpName(op_name_ + "_x")));
    }
    if (y->requires_grad()) {
      const auto& broadcast_mul_op =
          JUST(op_expr_helper::BroadcastMulOp(GradientOpName(op_name_ + "y_broadcast_mul")));
      const auto& broadcast_mul = JUST(Dispatch<Tensor>(*broadcast_mul_op, {out_grads.at(0), x}));
      in_grads->at(1) = JUST(ReduceSumLike(broadcast_mul, y, GradientOpName(op_name_ + "_y")));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_mul", BroadcastMul);

class BroadcastDiv : public BroadcastBinary {
 public:
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    JUST(BroadcastBinary::Capture(ctx, inputs, outputs));
    ctx->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    const auto& z = ctx->SavedTensors().at(2);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& broadcast_div_op =
          JUST(op_expr_helper::BroadcastDivOp(GradientOpName(op_name_ + "x_broadcast_div")));
      const auto& broadcast_div = JUST(Dispatch<Tensor>(*broadcast_div_op, {out_grads.at(0), y}));
      in_grads->at(0) = JUST(ReduceSumLike(broadcast_div, x, GradientOpName(op_name_ + "_x")));
    }
    if (y->requires_grad()) {
      const auto& broadcast_div_grad_op = JUST(op_expr_helper::BroadcastDivGradOp());
      in_grads->at(1) = JUST(Dispatch<Tensor>(*broadcast_div_grad_op, {out_grads.at(0), y, z}));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_div", BroadcastDiv);

}  // namespace one
}  // namespace oneflow
