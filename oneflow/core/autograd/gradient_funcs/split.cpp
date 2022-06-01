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
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct SplitCaptureState : public AutoGradCaptureState {
  int64_t axis;
  bool requires_grad;
};

class Split : public OpExprGradFunction<SplitCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SplitCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const SplitCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Split::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "Forward op must be not null";
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Split::Capture(SplitCaptureState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1) << "Input grad size must be equal 1";
  ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("dim"));
  return Maybe<void>::Ok();
}

Maybe<void> Split::Apply(const SplitCaptureState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  in_grads->resize(1);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  TensorTuple inputs;
  inputs.reserve(out_grads.size());
  for (int i = 0; i < out_grads.size(); ++i) {
    const auto& out_grad_i = JUST(VectorAt(out_grads, i));
    inputs.emplace_back(out_grad_i);
  }
  JUST(VectorAt(*in_grads, 0)) = JUST(functional::Concat(inputs, ctx->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("split", Split);

}  // namespace one
}  // namespace oneflow
