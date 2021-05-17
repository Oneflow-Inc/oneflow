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

struct PReLUInterpState : public OpExprInterpState {
  bool input_requires_grad;
  bool alpha_requires_grad;
};

class PReLU : public OpExprGradFunction<PReLUInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    const std::string& op_name = fw_op_expr->op_name();
    grad_op_ = JUST(op_expr_helper::PReLUGradOp(GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(PReLUInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();  // input
    ctx->alpha_requires_grad = inputs.at(1)->requires_grad();  // alpha
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const PReLUInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& dy = out_grads.at(0);
    const auto& x = ctx->SavedTensors().at(0);
    const auto& alpha = ctx->SavedTensors().at(1);

    in_grads->resize(2);
    if (ctx->input_requires_grad || ctx->alpha_requires_grad) {
      const auto& grads = JUST(OpInterpUtil::Dispatch<TensorTuple>(*grad_op_, {x, dy, alpha}));
      if (ctx->input_requires_grad) { in_grads->at(0) = grads->at(0); }
      if (ctx->alpha_requires_grad) { in_grads->at(1) = grads->at(1); }
    }

    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("prelu", PReLU);

}  // namespace one
}  // namespace oneflow
