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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SmoothL1LossInterpState : public OpExprInterpState {
  std::string reduction;
  float beta;
  size_t prediction_index;
  size_t label_index;
  bool requires_grad;
};

class SmoothL1Loss : public OpExprGradFunction<SmoothL1LossInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SmoothL1LossInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->requires_grad = inputs.at(0)->requires_grad();  // prediction
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->prediction_index = ctx->SaveTensorForBackward(inputs.at(0));  // prediction
    ctx->label_index = ctx->SaveTensorForBackward(inputs.at(1));       // label

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->beta = JUST(composed_attrs.GetAttr<float>("beta"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SmoothL1LossInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(2);

    if (ctx->requires_grad) {
      const auto& prediction = ctx->SavedTensors().at(ctx->prediction_index);
      const auto& label = ctx->SavedTensors().at(ctx->label_index);
      in_grads->at(0) =
          JUST(functional::SmoothL1LossGrad(out_grads.at(0), prediction, label, ctx->beta));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("smooth_l1_loss", SmoothL1Loss);  // todo: name

}  // namespace one
}  // namespace oneflow
