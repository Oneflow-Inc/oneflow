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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct PolygammaCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  int n = 1;
  int x_index = -1;
};

class Polygamma : public OpExprGradFunction<PolygammaCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(PolygammaCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const PolygammaCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Polygamma::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> Polygamma::Capture(PolygammaCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->n = JUST(composed_attrs.GetAttr<int>("n"));
  ctx->x_index = ctx->SaveTensorForBackward(inputs[0]);
  return Maybe<void>::Ok();
}

Maybe<void> Polygamma::Apply(const PolygammaCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& saved_tensors = ctx->SavedTensors();
  const auto& x = saved_tensors.at(ctx->x_index);
  auto polygamma_result = JUST(functional::Polygamma(ctx->n + 1, x));
  in_grads->at(0) = JUST(functional::Mul(out_grads.at(0), polygamma_result));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("polygamma", Polygamma);
}  // namespace one
}  // namespace oneflow
