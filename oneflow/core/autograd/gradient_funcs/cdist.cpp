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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct CDistCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  size_t x1_index = 0;
  size_t x2_index = 0;
  size_t out_index = 0;
  double p = 0.0;
};

class CDistGrad : public OpExprGradFunction<CDistCaptureState> {
 public:
  virtual ~CDistGrad() = default;

  using OpExprGradFunction<CDistCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(CDistCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const CDistCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> CDistGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> CDistGrad::Capture(CDistCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->x1_index = ctx->SaveTensorForBackward(inputs.at(0));
  ctx->x2_index = ctx->SaveTensorForBackward(inputs.at(1));
  ctx->out_index = ctx->SaveTensorForBackward(outputs.at(0));
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->p = JUST(composed_attrs.GetAttr<double>("p"));

  return Maybe<void>::Ok();
}

Maybe<void> CDistGrad::Apply(const CDistCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_LE_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  const auto& x1 = ctx->SavedTensors().at(ctx->x1_index);
  const auto& x2 = ctx->SavedTensors().at(ctx->x2_index);
  const auto& out = ctx->SavedTensors().at(ctx->out_index);
  const double p = ctx->p;

  in_grads->resize(2);
  auto results = JUST(functional::CDistGrad(x1, x2, out, out_grads.at(0), p));
  (*in_grads)[0] = results->at(0);
  (*in_grads)[1] = results->at(1);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_EXPR_GRAD_FUNCTION("cdist", CDistGrad);

}  // namespace one
}  // namespace oneflow
