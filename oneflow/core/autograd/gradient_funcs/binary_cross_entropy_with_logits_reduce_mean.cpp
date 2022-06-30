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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BinaryCrossEntropyWithLogitsReduceMeanCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  bool has_pos_weight = false;
};

class BinaryCrossEntropyWithLogitsReduceMean
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsReduceMeanCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx,
                      const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be null. ";
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Capture(
    BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx, const TensorTuple& inputs,
    const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // target
  return Maybe<void>::Ok();
}

Maybe<void> BinaryCrossEntropyWithLogitsReduceMean::Apply(
    const BinaryCrossEntropyWithLogitsReduceMeanCaptureState* ctx, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "out_grads size should be equal to 1. ";
  const auto& dy = JUST(VectorAt(out_grads, 0));
  const auto& input = JUST(VectorAt(ctx->SavedTensors(), 0));
  const auto& target = JUST(VectorAt(ctx->SavedTensors(), 1));
  in_grads->resize(ctx->SavedTensors().size());
  JUST(VectorAt(*in_grads, 0)) =
      JUST(functional::BinaryCrossEntropyWithLogitsReduceMeanLossGrad(dy, input, target));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits_reduce_mean",
                               BinaryCrossEntropyWithLogitsReduceMean);

}  // namespace one

}  // namespace oneflow
