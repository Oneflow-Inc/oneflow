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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/user_op_conf_trait.h"

namespace oneflow {
namespace one {

struct MatMulInterpState : public OpExprInterpState {
  bool transpose_a;
  bool transpose_b;
  bool requires_grad_a;
  bool requires_grad_b;

  // y = alpha * (a * b)
  double alpha;
};

class MatMul : public OpExprGradFunction<MatMulInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MatMulInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrValueMap& attrs) const override;
  Maybe<void> Apply(const MatMulInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> grad_a_op_;
  std::shared_ptr<OpExpr> grad_b_op_;
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
};

Maybe<void> MatMul::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  grad_a_op_ = JUST(op_expr_helper::MatMulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                             /*alpha=*/1.0, GradientOpName(op_name + "_a")));
  grad_b_op_ = JUST(op_expr_helper::MatMulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                             /*alpha=*/1.0, GradientOpName(op_name + "_b")));
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> MatMul::Capture(MatMulInterpState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  ctx->requires_grad_a = inputs.at(0)->requires_grad();
  ctx->requires_grad_b = inputs.at(1)->requires_grad();
  if (ctx->requires_grad_a) {
    ctx->SaveTensorForBackward(inputs.at(1));  // save b
  }
  if (ctx->requires_grad_b) {
    ctx->SaveTensorForBackward(inputs.at(0));  // save a
  }
  ctx->transpose_a = JUST(op_trait_->GetAttr<bool>("transpose_a", attrs));
  ctx->transpose_b = JUST(op_trait_->GetAttr<bool>("transpose_b", attrs));
  ctx->alpha = JUST(op_trait_->GetAttr<double>("alpha", attrs));
  return Maybe<void>::Ok();
}

Maybe<void> MatMul::Apply(const MatMulInterpState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  static auto dispatch = [](const OpExpr& matmul_op, const std::shared_ptr<Tensor>& a,
                            const std::shared_ptr<Tensor>& b, const bool& transpose_a,
                            const bool& transpose_b, const double& alpha) -> Maybe<Tensor> {
    AttrValueMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(matmul_op, {a, b}, attrs);
  };

  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    auto b = ctx->SavedTensors().at(0);
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(dispatch(*grad_a_op_, b, out_grads.at(0), ctx->transpose_b, true, ctx->alpha));
    } else {
      in_grads->at(0) =
          JUST(dispatch(*grad_a_op_, out_grads.at(0), b, false, !ctx->transpose_b, ctx->alpha));
    }
  }
  if (ctx->requires_grad_b) {
    auto a = ctx->SavedTensors().at(ctx->requires_grad_a);
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(dispatch(*grad_b_op_, out_grads.at(0), a, true, ctx->transpose_a, ctx->alpha));
    } else {
      in_grads->at(1) =
          JUST(dispatch(*grad_b_op_, a, out_grads.at(0), !ctx->transpose_a, false, ctx->alpha));
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", MatMul);

}  // namespace one
}  // namespace oneflow
