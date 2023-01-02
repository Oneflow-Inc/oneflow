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
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct LinalgCrossCaptureState : public AutoGradCaptureState {
  int64_t dim = -1;
  bool input_requires_grad = false;
  bool other_requires_grad = false;
};

class LinalgCross : public OpExprGradFunction<LinalgCrossCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(LinalgCrossCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const LinalgCrossCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> LinalgCross::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> LinalgCross::Capture(LinalgCrossCaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  ctx->other_requires_grad = inputs.at(1)->requires_grad();

  if (ctx->input_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
  if (ctx->other_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->dim = JUST(composed_attrs.GetAttr<int64_t>("dim"));
  return Maybe<void>::Ok();
}

Maybe<void> LinalgCross::Apply(const LinalgCrossCaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  in_grads->resize(ctx->SavedTensors().size());
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  if (ctx->input_requires_grad) {
    in_grads->at(0) =
        JUST(functional::LinalgCross(ctx->SavedTensors().at(0), out_grads.at(0), ctx->dim));
  }
  if (ctx->other_requires_grad) {
    in_grads->at(1) = JUST(functional::LinalgCross(
        out_grads.at(0), ctx->SavedTensors().at(ctx->input_requires_grad ? 1 : 0), ctx->dim));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("linalg_cross", LinalgCross);

}  // namespace one
}  // namespace oneflow