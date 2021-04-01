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
  std::shared_ptr<OpExpr> op(nullptr);
  TensorTuple inputs{input}, outputs(1);
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
  const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
  JUST(interpreter->Apply(*op, inputs, &outputs));
  return outputs.at(0);
}

Maybe<Tensor> ScalarMul(const std::shared_ptr<Tensor>& input, const float& scalar,
                        const std::string& op_name) {
  const auto& op = JUST(op_expr_helper::ScalarMulOp(scalar, op_name));
  const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
  TensorTuple outputs(1);
  JUST(interpreter->Apply(*op, {input}, &outputs));
  return outputs.at(0);
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
    x_requires_grad_ = inputs.at(0)->requires_grad();
    y_requires_grad_ = inputs.at(1)->requires_grad();
    x_shape_ = inputs.at(0)->shape();
    y_shape_ = inputs.at(1)->shape();
    z_shape_ = outputs.at(0)->shape();
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }

 protected:
  std::string op_name_;
  mutable bool x_requires_grad_;
  mutable bool y_requires_grad_;
  mutable std::shared_ptr<const Shape> x_shape_;
  mutable std::shared_ptr<const Shape> y_shape_;
  mutable std::shared_ptr<const Shape> z_shape_;
};

class BroadcastSub : public BroadcastBinary {
 public:
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x_requires_grad_) {
      in_grads->at(0) = JUST(ReduceSumLike(out_grads.at(0), x, GradientOpName(op_name_ + "_x")));
    }
    if (y_requires_grad_) {
      const auto& scalar_mul =
          JUST(ScalarMul(out_grads.at(0), -1.f, GradientOpName(op_name_ + "_y_scalar_mul")));
      in_grads->at(1) = JUST(ReduceSumLike(scalar_mul, y, GradientOpName(op_name_ + "_y")));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_sub", BroadcastSub);

}  // namespace one
}  // namespace oneflow
