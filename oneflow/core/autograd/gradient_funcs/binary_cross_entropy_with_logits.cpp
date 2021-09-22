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

struct BinaryCrossEntropyWithLogitsCaptureState : public AutoGradCaptureState {
  std::string reduction = "";
  bool has_pos_weight = false;
};

class BinaryCrossEntropyWithLogits
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> BinaryCrossEntropyWithLogits::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogits::Capture(BinaryCrossEntropyWithLogitsCaptureState* ctx,
                                                  const TensorTuple& inputs,
                                                  const TensorTuple& outputs,
                                                  const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->reduction = JUST(composed_attrs.GetAttr<std::string>("reduction"));
  ctx->has_pos_weight = JUST(composed_attrs.GetAttr<bool>("has_pos_weight"));
  ctx->SaveTensorForBackward(inputs.at(0));  // input
  ctx->SaveTensorForBackward(inputs.at(1));  // target
  if (inputs.size() == 3) {
    ctx->SaveTensorForBackward(inputs.at(2));  // weight or pos_weight
  }
  if (inputs.size() == 4) {
    ctx->SaveTensorForBackward(inputs.at(2));  // weight
    ctx->SaveTensorForBackward(inputs.at(3));  // pos_weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogits::Apply(const BinaryCrossEntropyWithLogitsCaptureState* ctx,
                                                const TensorTuple& out_grads,
                                                TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);
  const auto& input = ctx->SavedTensors().at(0);
  const auto& target = ctx->SavedTensors().at(1);

  in_grads->resize(ctx->SavedTensors().size());

  if (ctx->SavedTensors().size() == 3) {
    if (ctx->has_pos_weight) {
      const auto& pos_weight = ctx->SavedTensors().at(2);
      in_grads->at(0) = JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(
          dy, input, target, NullOpt, pos_weight, ctx->reduction));
    } else {
      const auto& weight = ctx->SavedTensors().at(2);
      in_grads->at(0) = JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(
          dy, input, target, weight, NullOpt, ctx->reduction));
    }
  } else if (ctx->SavedTensors().size() == 4) {
    const auto& weight = ctx->SavedTensors().at(2);
    const auto& pos_weight = ctx->SavedTensors().at(3);
    in_grads->at(0) = JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(
        dy, input, target, weight, pos_weight, ctx->reduction));
  } else {
    in_grads->at(0) = JUST(functional::BinaryCrossEntropyWithLogitsLossGrad(
        dy, input, target, NullOpt, NullOpt, ctx->reduction));
  }
  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits", BinaryCrossEntropyWithLogits);
}  // namespace one
}  // namespace oneflow
