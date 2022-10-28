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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct InvCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
};

class Inv : public OpExprGradFunction<InvCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(InvCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
    if (ctx->requires_grad) { ctx->SaveTensorForBackward(JUST(VectorAt(outputs, 0))); }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const InvCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (ctx->requires_grad) {
      const auto& output = JUST(VectorAt(ctx->SavedTensors(), 0));
      const auto& dy = JUST(VectorAt(out_grads, 0));
      JUST(VectorAt(*in_grads, 0)) = JUST(functional::Negative(JUST(functional::MatMul(
          output, JUST(functional::MatMul(dy, output, false, true, 1.0)), true, false, 1.0))));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("inv", Inv);

}  // namespace one
}  // namespace oneflow
