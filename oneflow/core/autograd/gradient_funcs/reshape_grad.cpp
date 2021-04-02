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

class ReshapeOpExprGrad : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    backward_op_ = JUST(op_expr_helper::ReshapeLikeOp(GradientOpName(fw_op_expr->op_name())));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& saved_tensors = ctx->SavedTensors();
    const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
    in_grads->resize(1);
    JUST(interpreter->Apply(*backward_op_, {out_grads.at(0), saved_tensors.at(0)}, in_grads));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> backward_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reshape", ReshapeOpExprGrad);

}  // namespace one
}  // namespace oneflow
