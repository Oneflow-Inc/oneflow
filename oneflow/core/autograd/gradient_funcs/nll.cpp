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
struct NllCaptureState : public AutoGradCaptureState {
  int64_t ignore_index = -100;
  std::string reduction = "";
};

class Nll : public OpExprGradFunction<NllCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(NllCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const NllCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};
Maybe<void> Nll::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
Maybe<void> Nll::Capture(NllCaptureState* ctx, const TensorTuple& inputs,
                         const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->ignore_index = JUST(composed_attrs.GetAttr<int64_t>("ignore_index"));
  ctx->reduction = JUST(composed_attrs.GetAttr<std::string>("reduction"));
  ctx->SaveTensorForBackward(inputs.at(0));   // input
  ctx->SaveTensorForBackward(inputs.at(1));   // target
  ctx->SaveTensorForBackward(outputs.at(1));  // total_weight
  if (inputs.size() == 3) {
    ctx->SaveTensorForBackward(inputs.at(2));  // weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> Nll::Apply(const NllCaptureState* ctx, const TensorTuple& out_grads,
                       TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);
  const auto& dy = out_grads.at(0);
  const auto& input = ctx->SavedTensors().at(0);
  const auto& target = ctx->SavedTensors().at(1);
  const auto& total_weight = ctx->SavedTensors().at(2);

  in_grads->resize(ctx->SavedTensors().size() - 1);

  if (ctx->SavedTensors().size() == 4) {
    const auto& weight = ctx->SavedTensors().at(3);
    in_grads->at(0) = JUST(functional::NllLossGrad(dy, input, target, weight, total_weight,
                                                   ctx->ignore_index, ctx->reduction));
  } else {
    in_grads->at(0) = JUST(functional::NllLossGrad(dy, input, target, NullOpt, total_weight,
                                                   ctx->ignore_index, ctx->reduction));
  }
  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("nll", Nll);
}  // namespace one
}  // namespace oneflow
