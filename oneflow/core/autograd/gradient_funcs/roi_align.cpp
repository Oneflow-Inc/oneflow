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

struct RoiAlignCaptureState : public AutoGradCaptureState {
  float spatial_scale = 1.0;
  int32_t pooled_h = 0;
  int32_t pooled_w = 0;
  int32_t sampling_ratio = -1;
  bool aligned = false;
  bool requires_grad = false;
};

class RoiAlign : public OpExprGradFunction<RoiAlignCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(RoiAlignCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->spatial_scale = JUST(composed_attrs.GetAttr<float>("spatial_scale"));
    ctx->pooled_h = JUST(composed_attrs.GetAttr<int32_t>("pooled_h"));
    ctx->pooled_w = JUST(composed_attrs.GetAttr<int32_t>("pooled_w"));
    ctx->sampling_ratio = JUST(composed_attrs.GetAttr<int32_t>("sampling_ratio"));
    ctx->aligned = JUST(composed_attrs.GetAttr<bool>("aligned"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const RoiAlignCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    const auto& x_like = ctx->SavedTensors().at(0);
    const auto& rois = ctx->SavedTensors().at(1);
    in_grads->at(0) = JUST(
        functional::RoiAlignGrad(out_grads.at(0), x_like, rois, ctx->spatial_scale, ctx->pooled_h,
                                 ctx->pooled_w, ctx->sampling_ratio, ctx->aligned));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("roi_align", RoiAlign);

}  // namespace one
}  // namespace oneflow
