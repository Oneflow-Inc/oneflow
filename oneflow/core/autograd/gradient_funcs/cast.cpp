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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {
namespace one {

struct CastCaptureState : public AutoGradCaptureState {
  Symbol<DType> dtype;
};

class Cast : public OpExprGradFunction<CastCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CastCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ctx->dtype = inputs.at(0)->dtype();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CastCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    (*in_grads)[0] = JUST(functional::Cast(out_grads[0], ctx->dtype, /*pin_memory=*/false));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("cast", Cast);

}  // namespace one
}  // namespace oneflow
