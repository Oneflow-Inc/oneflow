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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ReshapeCaptureState : public AutoGradCaptureState {
  DimVector input_shape_vec;
};

class ReshapeGrad : public OpExprGradFunction<ReshapeCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReshapeCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->input_shape_vec = inputs.at(0)->shape()->dim_vec();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReshapeCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    Shape shape(ctx->input_shape_vec);
    in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), shape));
    return Maybe<void>::Ok();
  }
};

class ReshapeLikeGrad : public OpExprGradFunction<ReshapeCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReshapeCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
    CHECK_OR_RETURN(!inputs.at(1)->requires_grad())
        << "ReshapeLikeOp's input[1] need not requires_grad.";
    ctx->input_shape_vec = inputs.at(0)->shape()->dim_vec();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReshapeCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    Shape shape(ctx->input_shape_vec);
    in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), shape));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reshape", ReshapeGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("reshape_like", ReshapeLikeGrad);

}  // namespace one
}  // namespace oneflow
