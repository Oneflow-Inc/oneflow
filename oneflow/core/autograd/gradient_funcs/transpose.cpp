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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct TransposeCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> perm;
  bool requires_grad;
};

class Transpose : public OpExprGradFunction<TransposeCaptureState> {
 public:
  Maybe<void> Capture(TransposeCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const TransposeCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Transpose::Capture(TransposeCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* op_ctx = dynamic_cast<const TransposeOp*>(ctx);
  state->perm = op_ctx->perm();
  return Maybe<void>::Ok();
}

Maybe<void> Transpose::Apply(const TransposeCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  std::vector<int32_t> grad_perm;
  grad_perm.resize(state->perm.size());
  FOR_RANGE(int32_t, i, 0, state->perm.size()) { grad_perm.at(state->perm.at(i)) = i; }
  in_grads->at(0) = JUST(functional::Transpose(out_grads.at(0), grad_perm));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("transpose", Transpose);

}  // namespace one
}  // namespace oneflow
