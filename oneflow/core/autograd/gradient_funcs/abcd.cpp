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

struct AbcdExprInterpState : public OpExprInterpState {
  bool requires_grad;
};

class Abcd : public OpExprGradFunction<AbcdExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { 
    return Maybe<void>::Ok(); }

  Maybe<void> Capture(AbcdExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const AbcdExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    grad_op_ = JUST(op_expr_helper::ZerosOp());
    in_grads->at(0) = out_grads.at(0);
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("abcd", Abcd);

}  // namespace one
}  // namespace oneflow
