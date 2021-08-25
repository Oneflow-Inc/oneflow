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

struct GirdSampleInterpState : public AutoGradCaptureState {
  Shape size;
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
  size_t input_index;
  size_t gird_index;
  bool input_requires_grad;
  bool gird_requires_grad;
  bool requires_grad;
};

class GirdSample : public OpExprGradFunction<GirdSampleInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(GirdSampleInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();
    ctx->gird_requires_grad = inputs.at(1)->requires_grad();
    ctx->requires_grad = ctx->input_requires_grad || ctx->gird_requires_grad;
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->input_index = ctx->SaveTensorForBackward(inputs.at(0));  // input
    ctx->gird_index = ctx->SaveTensorForBackward(inputs.at(1));   // gird

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->interpolation_mode = JUST(composed_attrs.GetAttr<std::string>("interpolation_mode"));
    ctx->padding_mode = JUST(composed_attrs.GetAttr<std::string>("padding_mode"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const GirdSampleInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);

    if (ctx->requires_grad) {
      const auto& input = ctx->SavedTensors().at(ctx->input_index);
      const auto& gird = ctx->SavedTensors().at(ctx->gird_index);
      const auto& results =
          JUST(functional::GridSampleGrad(out_grads.at(0), input, gird, ctx->interpolation_mode,
                                          ctx->padding_mode, ctx->align_corners));

      in_grads->resize(2);
      if (ctx->input_requires_grad) { in_grads->at(0) = results->at(0); }
      if (ctx->gird_requires_grad) { in_grads->at(1) = results->at(1); }
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("grid_sample", GirdSample);

}  // namespace one
}  // namespace oneflow
