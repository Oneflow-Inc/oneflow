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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct StackCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
  int64_t axis = 1;
  int64_t input_num = 2;
};

class Stack : public OpExprGradFunction<StackCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(StackCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const StackCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Stack::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Stack::Capture(StackCaptureState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad.resize(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) { ctx->requires_grad[i] = inputs.at(i)->requires_grad(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("axis"));
  for (const auto& input : inputs) { ctx->SaveTensorForBackward(input); }
  ctx->input_num = inputs.size();
  return Maybe<void>::Ok();
}

Maybe<void> Stack::Apply(const StackCaptureState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(ctx->input_num);
  TensorTuple like(ctx->input_num);
  for (int i = 0; i < ctx->input_num; ++i) { like[i] = ctx->SavedTensors().at(i); }
  const auto& results = JUST(functional::StackGrad(out_grads.at(0), like, ctx->axis));
  CHECK_EQ_OR_RETURN(results->size(), ctx->input_num)
      << Error::RuntimeError() << "The number of results (" << results->size()
      << ") must match the number of inputs (" << ctx->input_num << ")";
  for (int i = 0; i < ctx->input_num; ++i) {
    if (ctx->requires_grad.at(i)) { in_grads->at(i) = results->at(i); }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("stack", Stack);

}  // namespace one
}  // namespace oneflow
