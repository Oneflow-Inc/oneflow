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
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {
namespace one {

struct IdentityCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class Identity : public OpExprGradFunction<IdentityCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(IdentityCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const IdentityCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(1);
    if (ctx->requires_grad) {
      if (LazyMode::is_enabled()) {
        // requires an intermediate node to avoid redundant memory copy or commnet
        // communication in lazy mode
        in_grads->at(0) = JUST(functional::Identity(out_grads.at(0)));
      } else {
        in_grads->at(0) = out_grads.at(0);
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("identity", Identity);

}  // namespace one
}  // namespace oneflow
