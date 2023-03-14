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
#include <string>
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct TransposedBinaryOpCaptureState : public AutoGradCaptureState {
  bool lhs_requires_grad = false;
  bool rhs_requires_grad = false;
  std::string op;
  bool inplace;
};

class TransposedBinaryOp : public OpExprGradFunction<TransposedBinaryOpCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TransposedBinaryOpCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TransposedBinaryOpCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> TransposedBinaryOp::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> TransposedBinaryOp::Capture(TransposedBinaryOpCaptureState* ctx,
                                        const TensorTuple& inputs, const TensorTuple& outputs,
                                        const AttrMap& attrs) const {
  ctx->lhs_requires_grad = inputs.at(0)->requires_grad();
  ctx->rhs_requires_grad = inputs.at(1)->requires_grad();
  if (!ctx->lhs_requires_grad && !ctx->rhs_requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
  ctx->op = JUST(composed_attrs.GetAttr<std::string>("op"));
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));
  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> TransposedBinaryOp::Apply(const TransposedBinaryOpCaptureState* ctx,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->lhs_requires_grad && !ctx->rhs_requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  auto lhs = ctx->SavedTensors().at(0);
  auto rhs = ctx->SavedTensors().at(1);
  auto y = ctx->SavedTensors().at(2);
  auto ret = JUST(functional::TransposedBinaryOpGrad(out_grads.at(0), y, lhs, rhs, ctx->op, false));
  if (ctx->lhs_requires_grad) in_grads->at(0) = ret->at(0);
  if (ctx->rhs_requires_grad) in_grads->at(1) = ret->at(1);
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("transposed_binary_op", TransposedBinaryOp);

}  // namespace one
}  // namespace oneflow
