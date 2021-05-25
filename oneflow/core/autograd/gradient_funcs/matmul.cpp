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

namespace oneflow {
namespace one {

struct MatmulInterpState : public OpExprInterpState {
  bool transpose_a;
  bool transpose_b;
  double alpha;
  bool requires_grad_a;
  bool requires_grad_b;
};

class Matmul : public OpExprGradFunction<MatmulInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MatmulInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const MatmulInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_a_op1_;
  std::shared_ptr<OpExpr> grad_a_op2_;
  std::shared_ptr<OpExpr> grad_b_op1_;
  std::shared_ptr<OpExpr> grad_b_op2_;
};

Maybe<void> Matmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  bool transpose_a;
  bool transpose_b;
  double alpha;
  grad_a_op1_ =
      JUST(op_expr_helper::MatmulGradOp(transpose_b, true, alpha, GradientOpName(op_name)));
  grad_a_op2_ =
      JUST(op_expr_helper::MatmulGradOp(false, !transpose_b, alpha, GradientOpName(op_name)));
  grad_b_op1_ =
      JUST(op_expr_helper::MatmulGradOp(true, transpose_a, alpha, GradientOpName(op_name)));
  grad_b_op2_ =
      JUST(op_expr_helper::MatmulGradOp(!transpose_a, false, alpha, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Capture(MatmulInterpState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_a = inputs.at(0)->requires_grad();
  ctx->requires_grad_b = inputs.at(1)->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->transpose_a = JUST(composed_attrs.GetAttr<bool>("transpose_a"));
  ctx->transpose_b = JUST(composed_attrs.GetAttr<bool>("transpose_b"));
  ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
  ctx->SaveTensorForBackward(inputs.at(0));  // input a
  ctx->SaveTensorForBackward(inputs.at(1));  // input b
  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Apply(const MatmulInterpState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  MutableAttrMap attrs;

  const auto& input_a = ctx->SavedTensors().at(0);
  const auto& input_b = ctx->SavedTensors().at(1);
  if (ctx->requires_grad_a) {
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_a_op1_, {input_b, out_grads.at(0)}, attrs));
    } else {
      in_grads->at(0) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_a_op2_, {out_grads.at(0), input_b}, attrs));
    }
  }

  if (ctx->requires_grad_b) {
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op1_, {out_grads.at(0), input_a}, attrs));
    } else {
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op2_, {input_a, out_grads.at(0)}, attrs));
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", Matmul);

}  // namespace one
}  // namespace oneflow
