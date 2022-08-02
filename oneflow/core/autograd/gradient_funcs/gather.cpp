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

struct GatherCaptureState : public AutoGradCaptureState {
  int64_t axis;
  bool requires_grad;
};

class Gather : public OpExprGradFunction<GatherCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GatherCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GatherCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Gather::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Gather::Capture(GatherCaptureState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("axis"));
  return Maybe<void>::Ok();
}

Maybe<void> Gather::Apply(const GatherCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& x = ctx->SavedTensors().at(0);
  const auto& indices = ctx->SavedTensors().at(1);
  in_grads->at(0) =
      JUST(functional::UnsortedSegmentSumLike(out_grads.at(0), indices, x, ctx->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("gather", Gather);

}  // namespace one
}  // namespace oneflow
