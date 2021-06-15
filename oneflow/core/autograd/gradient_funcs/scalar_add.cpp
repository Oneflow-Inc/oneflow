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

namespace oneflow {
namespace one {

struct ScalarAddInterpState : public OpExprInterpState {
  bool requires_grad;
};

class ScalarAdd : public OpExprGradFunction<ScalarAddInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(ScalarAddInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarAddInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) { in_grads->at(0) = out_grads.at(0); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_add", ScalarAdd);

}  // namespace one
}  // namespace oneflow
