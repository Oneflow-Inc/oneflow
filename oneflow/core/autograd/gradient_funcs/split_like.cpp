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

struct SplitLikeCaptureState : public AutoGradCaptureState {
  int64_t axis;
  bool requires_grad;
};

class SplitLike : public OpExprGradFunction<SplitLikeCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SplitLikeCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SplitLikeCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> SplitLike::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Capture(SplitLikeCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), outputs.size() + 1);  // NOLINT(maybe-need-error-msg)
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("axis"));
  for (int i = 0; i < outputs.size(); ++i) { ctx->SaveTensorForBackward(outputs.at(i)); }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Apply(const SplitLikeCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  in_grads->resize(out_grads.size() + 1);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  const auto& saved_tensors = ctx->SavedTensors();
  TensorTuple inputs;
  inputs.reserve(out_grads.size());
  for (int i = 0; i < out_grads.size(); ++i) {
    const auto& out_grad_i = out_grads.at(i);
    if (out_grad_i.get()) {
      inputs.emplace_back(out_grad_i);
    } else {
      const auto& zero_grad = JUST(functional::ZerosLike(saved_tensors.at(i)));
      inputs.emplace_back(zero_grad);
    }
  }
  in_grads->at(0) = JUST(functional::Concat(inputs, ctx->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("split_like", SplitLike);

}  // namespace one
}  // namespace oneflow
