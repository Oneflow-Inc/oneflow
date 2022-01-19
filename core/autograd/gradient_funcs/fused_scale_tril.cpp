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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedScaleTrilState : public AutoGradCaptureState {
  bool requires_grad;
  int64_t diagonal;
  double floating_scale_value;
  int64_t integer_scale_value;
  bool is_floating_scale_value;
};

class FusedScaleTril : public OpExprGradFunction<FusedScaleTrilState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedScaleTrilState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedScaleTrilState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedScaleTril::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleTril::Capture(FusedScaleTrilState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->diagonal = JUST(composed_attrs.GetAttr<int64_t>("diagonal"));
  ctx->floating_scale_value = JUST(composed_attrs.GetAttr<double>("floating_scale_value"));
  ctx->integer_scale_value = JUST(composed_attrs.GetAttr<int64_t>("integer_scale_value"));
  ctx->is_floating_scale_value = JUST(composed_attrs.GetAttr<bool>("is_floating_scale_value"));
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleTril::Apply(const FusedScaleTrilState* ctx, const TensorTuple& out_grads,
                                  TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  Scalar scale;
  if (ctx->is_floating_scale_value) {
    scale = ctx->floating_scale_value;
  } else {
    scale = ctx->integer_scale_value;
  }
  (*in_grads)[0] = JUST(functional::FusedScaleTril(out_grads[0], ctx->diagonal, 0, scale));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_tril", FusedScaleTril);

}  // namespace one
}  // namespace oneflow
