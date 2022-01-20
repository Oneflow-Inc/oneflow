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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SelectCaptureState : public AutoGradCaptureState {
  int32_t dim = 0;
  int32_t index = 0;
  bool requires_grad = false;
};

class Select : public OpExprGradFunction<SelectCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SelectCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SelectCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override ;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Select::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Select::Capture(SelectCaptureState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->dim = JUST(composed_attrs.GetAttr<int32_t>("dim"));
  ctx->index = JUST(composed_attrs.GetAttr<int32_t>("index"));
  return Maybe<void>::Ok();
}

Maybe<void> Select::Apply(const SelectCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const auto& input = ctx->SavedTensors().at(0);
  int32_t dim = ctx->dim;
  int32_t index = ctx->index;

  in_grads->at(0) =
      JUST(functional::SelectGrad(out_grads.at(0), input, dim, index));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("select", Select);

}  // namespace one
}  // namespace oneflow