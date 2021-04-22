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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

namespace {

class ReduceSumLikeModule {
 public:
  ReduceSumLikeModule(const std::string& op_name) {
    identity_op_ = op_expr_helper::IdentityOp(op_name).GetPtrOrThrow();
    reshape_like_op_ = op_expr_helper::ReshapeLikeOp(op_name).GetPtrOrThrow();
    reduce_sum_like_op_ = op_expr_helper::ReduceSumLikeOp({-1}, op_name).GetPtrOrThrow();
  }

  Maybe<Tensor> forward(const std::shared_ptr<Tensor>& input,
                        const std::shared_ptr<Tensor>& like) const {
    const auto& in_shape = *(input->shape());
    const auto& like_shape = *(like->shape());
    TensorTuple inputs{input};
    AttrValueMap attrs;
    std::shared_ptr<OpExpr> op = identity_op_;
    if (in_shape != like_shape) {
      const Shape& left_extended_shape =
          CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
      if (in_shape == left_extended_shape) {
        op = reshape_like_op_;
      } else {
        op = reduce_sum_like_op_;
        const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
        JUST(attrs.SetAttr<std::vector<int32_t>>(
            "axis", std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()}));
      }
      inputs.push_back(like);
    }
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op, inputs, attrs));
  }

 private:
  std::shared_ptr<OpExpr> identity_op_;
  std::shared_ptr<OpExpr> reshape_like_op_;
  std::shared_ptr<OpExpr> reduce_sum_like_op_;
};

}  // namespace

class BroadcastBinary : public OpExprGradFunction<OpExprInterpState> {
 public:
  BroadcastBinary() = default;
  virtual ~BroadcastBinary() = default;

  virtual Maybe<void> Init(const OpExpr& op) {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    op_name_ = fw_op_expr->op_name();
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrValueMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->SaveTensorForBackward(inputs.at(0)->detach());
    ctx->SaveTensorForBackward(inputs.at(1)->detach());
    ctx->SaveTensorForBackward(outputs.at(0)->detach());
    return Maybe<void>::Ok();
  }

 protected:
  std::string op_name_;
};

class BroadcastAdd : public BroadcastBinary {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastBinary::Init(op));
    x_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_x");
    y_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_y");
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) { in_grads->at(0) = JUST(x_grad_op_->forward(out_grads.at(0), x)); }
    if (y->requires_grad()) { in_grads->at(1) = JUST(y_grad_op_->forward(out_grads.at(0), y)); }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<ReduceSumLikeModule> x_grad_op_;
  std::shared_ptr<ReduceSumLikeModule> y_grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_add", BroadcastAdd);

class BroadcastSub : public BroadcastBinary {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastBinary::Init(op));
    x_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_x");
    y_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_y");
    y_grad_mul_op_ =
        JUST(op_expr_helper::ScalarMulOp(-1.f, GradientOpName(op_name_ + "_y_scalar_mul")));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) { in_grads->at(0) = JUST(x_grad_op_->forward(out_grads.at(0), x)); }
    if (y->requires_grad()) {
      const auto& grad =
          JUST(OpInterpUtil::Dispatch<Tensor>(*y_grad_mul_op_, {out_grads.at(0)}, /*attrs=*/{}));
      in_grads->at(1) = JUST(y_grad_op_->forward(grad, y));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<ReduceSumLikeModule> x_grad_op_;
  std::shared_ptr<ReduceSumLikeModule> y_grad_op_;
  std::shared_ptr<OpExpr> y_grad_mul_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_sub", BroadcastSub);

class BroadcastMul : public BroadcastBinary {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastBinary::Init(op));
    x_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_x");
    y_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_y");
    x_grad_mul_op_ =
        JUST(op_expr_helper::BroadcastMulOp(GradientOpName(op_name_ + "_x_broadcast_mul")));
    y_grad_mul_op_ =
        JUST(op_expr_helper::BroadcastMulOp(GradientOpName(op_name_ + "_y_broadcast_mul")));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& x_grad =
          JUST(OpInterpUtil::Dispatch<Tensor>(*x_grad_mul_op_, {out_grads.at(0), y}, /*attrs=*/{}));
      in_grads->at(0) = JUST(x_grad_op_->forward(x_grad, x));
    }
    if (y->requires_grad()) {
      const auto& y_grad =
          JUST(OpInterpUtil::Dispatch<Tensor>(*y_grad_mul_op_, {out_grads.at(0), x}, /*attrs=*/{}));
      in_grads->at(1) = JUST(y_grad_op_->forward(y_grad, y));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<ReduceSumLikeModule> x_grad_op_;
  std::shared_ptr<ReduceSumLikeModule> y_grad_op_;
  std::shared_ptr<OpExpr> x_grad_mul_op_;
  std::shared_ptr<OpExpr> y_grad_mul_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_mul", BroadcastMul);

class BroadcastDiv : public BroadcastBinary {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastBinary::Init(op));
    x_grad_op_ = std::make_shared<ReduceSumLikeModule>(op_name_ + "_x");
    x_grad_div_op_ =
        JUST(op_expr_helper::BroadcastDivOp(GradientOpName(op_name_ + "_x_broadcast_div")));
    y_grad_op_ = JUST(op_expr_helper::BroadcastDivGradOp(GradientOpName(op_name_ + "_y")));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    const auto& z = ctx->SavedTensors().at(2);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& x_grad =
          JUST(OpInterpUtil::Dispatch<Tensor>(*x_grad_div_op_, {out_grads.at(0), y}, /*attrs=*/{}));
      in_grads->at(0) = JUST(x_grad_op_->forward(x_grad, x));
    }
    if (y->requires_grad()) {
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*y_grad_op_, {out_grads.at(0), y, z}, /*attrs=*/{}));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<ReduceSumLikeModule> x_grad_op_;
  std::shared_ptr<OpExpr> x_grad_div_op_;
  std::shared_ptr<OpExpr> y_grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_div", BroadcastDiv);

}  // namespace one
}  // namespace oneflow
