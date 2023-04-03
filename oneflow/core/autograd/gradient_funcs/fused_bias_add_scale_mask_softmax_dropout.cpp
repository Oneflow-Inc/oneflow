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

namespace oneflow {
namespace one {

struct FusedBiasAddScaleMaskSoftmaxDropoutCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool bias_requires_grad = false;
  bool bias_broadcast = false;
  int softmax_y_index = -1;
  int bias_index = -1;
  int mask_index = -1;
  int dropout_mask_index = -1;
  float scale = 1.0;
  float dropout_scale = 1.0;
};

class FusedBiasAddScaleMaskSoftmaxDropoutGradFunction
    : public OpExprGradFunction<FusedBiasAddScaleMaskSoftmaxDropoutCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedBiasAddScaleMaskSoftmaxDropoutCaptureState* ctx,
                      const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(outputs.size(), 2);  // (y, softmax_y)
    CHECK_EQ_OR_RETURN(inputs.size(), 4);   // (x, bias, mask, dropout_mask)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->bias_requires_grad = inputs.at(1)->requires_grad();

    if (!ctx->x_requires_grad && !ctx->bias_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale = JUST(composed_attrs.GetAttr<float>("scale_value"));
    ctx->dropout_scale = JUST(composed_attrs.GetAttr<float>("dropout_scale_value"));

    if (ctx->x_requires_grad) {
      ctx->mask_index = ctx->SaveTensorForBackward(inputs.at(2));          // mask
      ctx->dropout_mask_index = ctx->SaveTensorForBackward(inputs.at(3));  // dropout_mask
      ctx->softmax_y_index = ctx->SaveTensorForBackward(outputs.at(1));    // softmax_y
    }

    if (ctx->bias_requires_grad) {
      ctx->bias_broadcast = (inputs.at(0)->shape() != inputs.at(1)->shape());
      if (ctx->bias_broadcast) {
        ctx->bias_index = ctx->SaveTensorForBackward(inputs.at(1));  // bias
      }
    }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedBiasAddScaleMaskSoftmaxDropoutCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad && !ctx->bias_requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // (dy, d_softmax_y)
    in_grads->resize(4);                      // (x, bias, mask, dropout_mask)

    const auto& saved_tensors = ctx->SavedTensors();
    const auto& dy = out_grads.at(0);
    CHECK_GE_OR_RETURN(saved_tensors.size(), 3);  // (mask, dropout_mask, softmax_y, [bias])

    if (ctx->x_requires_grad || ctx->bias_requires_grad) {
      const auto& mask = saved_tensors.at(ctx->mask_index);
      const auto& dropout_mask = saved_tensors.at(ctx->dropout_mask_index);
      const auto& softmax_y = saved_tensors.at(ctx->softmax_y_index);
      in_grads->at(0) = JUST(functional::FusedScaleMaskSoftmaxDropoutGrad(
          softmax_y, dy, mask, dropout_mask, ctx->scale, ctx->dropout_scale));
    }

    if (ctx->bias_requires_grad) {
      if (ctx->bias_broadcast) {
        const auto& bias = saved_tensors.at(ctx->bias_index);
        in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(in_grads->at(0), bias));
      } else {
        in_grads->at(1) = in_grads->at(0);
      }
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_bias_add_scale_mask_softmax_dropout",
                               FusedBiasAddScaleMaskSoftmaxDropoutGradFunction);

}  // namespace one
}  // namespace oneflow
