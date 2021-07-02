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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

struct ReturnFirstInputExprInterpState : public OpExprInterpState {
  TensorTuple inputs;
  bool requires_grad;
};

class ReturnFirstInput : public OpExprGradFunction<ReturnFirstInputExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(ReturnFirstInputExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->inputs = inputs;
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReturnFirstInputExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->at(0) = out_grads.at(0);
    for (int i = 1; i < ctx->inputs.size(); i++) {
      const auto& tensor = ctx->inputs.at(i);
      std::shared_ptr<OpExpr> grad_op = JUST(op_expr_helper::ZeroLikeOp());
      in_grads->at(i) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op, {tensor}, {}));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("return_first_input", ReturnFirstInput);

}  // namespace one
}  // namespace oneflow
