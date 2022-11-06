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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedScaleMaskSoftmaxDropoutInterState : public AutoGradCaptureState {
  bool input_requires_grad = true;
  float scale = 1.0;
  float dropout_scale = 1.0;
};

class FusedScaleMaskSoftmaxDropout
    : public OpExprGradFunction<FusedScaleMaskSoftmaxDropoutInterState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedScaleMaskSoftmaxDropoutInterState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedScaleMaskSoftmaxDropoutInterState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedScaleMaskSoftmaxDropout::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmaxDropout::Capture(FusedScaleMaskSoftmaxDropoutInterState* ctx,
                                                  const TensorTuple& inputs,
                                                  const TensorTuple& outputs,
                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // input, mask, dropout_mask
  ctx->input_requires_grad = inputs.at(0)->requires_grad();

  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->scale = JUST(composed_attrs.GetAttr<float>("scale_value"));
  ctx->dropout_scale = JUST(composed_attrs.GetAttr<float>("dropout_scale_value"));

  ctx->SaveTensorForBackward(inputs.at(1));   // mask
  ctx->SaveTensorForBackward(inputs.at(2));   // dropout_mask
  ctx->SaveTensorForBackward(outputs.at(1));  // softmax_y
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmaxDropout::Apply(const FusedScaleMaskSoftmaxDropoutInterState* ctx,
                                                const TensorTuple& out_grads,
                                                TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // dy, d_softmax_y
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(3);  // input, mask, dropout_mask

  const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& dropout_mask = ctx->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& softmax_y = ctx->SavedTensors().at(2);
  const std::shared_ptr<oneflow::one::Tensor>& input_grad =
      JUST(functional::FusedScaleMaskSoftmaxDropoutGrad(
          softmax_y, out_grads.at(0), mask, dropout_mask, ctx->scale, ctx->dropout_scale));

  in_grads->at(0) = input_grad;
  return Maybe<void>::Ok();
}

struct FusedBiasAddScaleMaskSoftmaxDropoutCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool bias_requires_grad = false;
  bool bias_broadcast = false;
  int bias_index = 0;
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
    CHECK_EQ_OR_RETURN(inputs.size(), 4);  // (x, bias, mask, dropout_mask)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->bias_requires_grad = inputs.at(1)->requires_grad();

    if (!ctx->x_requires_grad && !ctx->bias_requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale = JUST(composed_attrs.GetAttr<float>("scale_value"));
    ctx->dropout_scale = JUST(composed_attrs.GetAttr<float>("dropout_scale_value"));

    if (ctx->x_requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(2));   // mask
      ctx->SaveTensorForBackward(inputs.at(3));   // dropout_mask
      ctx->SaveTensorForBackward(outputs.at(1));  // softmax_y
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

    if (ctx->x_requires_grad) {
      CHECK_GE_OR_RETURN(saved_tensors.size(), 3);  // (mask, dropout_mask, softmax_y, [bias])
      const auto& mask = saved_tensors.at(0);
      const auto& dropout_mask = saved_tensors.at(1);
      const auto& softmax_y = saved_tensors.at(2);
      in_grads->at(0) = JUST(functional::FusedScaleMaskSoftmaxDropoutGrad(
          softmax_y, dy, mask, dropout_mask, ctx->scale, ctx->dropout_scale));
    }

    if (ctx->bias_requires_grad) {
      if (ctx->bias_broadcast) {
        const auto& bias = saved_tensors.at(ctx->bias_index);
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(dy, bias));
      } else {
        in_grads->at(0) = dy;
      }
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_mask_softmax_dropout", FusedScaleMaskSoftmaxDropout);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_bias_add_scale_mask_softmax_dropout",
                               FusedBiasAddScaleMaskSoftmaxDropoutGradFunction);

}  // namespace one
}  // namespace oneflow
