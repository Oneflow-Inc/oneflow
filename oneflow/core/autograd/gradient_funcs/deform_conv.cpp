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

struct DeformConvNdCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool offset_requires_grad = false;
  bool weight_requires_grad = false;
  bool mask_requires_grad = false;
  bool bias_requires_grad = false;
  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t pad_h = 0;
  int32_t pad_w = 0;
  int32_t dilation_h = 0;
  int32_t dilation_w = 0;
  int32_t groups = 0;
  int32_t offset_groups = 0;
  bool use_mask = false;
};

class DeformConvNd : public OpExprGradFunction<DeformConvNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DeformConvNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DeformConvNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> DeformConvNd::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> DeformConvNd::Capture(DeformConvNdCaptureState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  ctx->weight_requires_grad = inputs.at(1)->requires_grad();
  ctx->offset_requires_grad = inputs.at(2)->requires_grad();
  ctx->mask_requires_grad = inputs.at(3)->requires_grad();

  ctx->SaveTensorForBackward(inputs.at(0));  // input
  ctx->SaveTensorForBackward(inputs.at(1));  // weight
  ctx->SaveTensorForBackward(inputs.at(2));  // offset
  ctx->SaveTensorForBackward(inputs.at(3));  // mask

  ComposedAttrMap composed_attrs(attrs, base_attrs_);

  ctx->use_mask = JUST(composed_attrs.GetAttr<bool>("use_mask"));
  ctx->stride_h = JUST(composed_attrs.GetAttr<int32_t>("stride_h"));
  ctx->stride_w = JUST(composed_attrs.GetAttr<int32_t>("stride_w"));
  ctx->pad_h = JUST(composed_attrs.GetAttr<int32_t>("pad_h"));
  ctx->pad_w = JUST(composed_attrs.GetAttr<int32_t>("pad_w"));
  ctx->dilation_h = JUST(composed_attrs.GetAttr<int32_t>("dilation_h"));
  ctx->dilation_w = JUST(composed_attrs.GetAttr<int32_t>("dilation_w"));
  ctx->groups = JUST(composed_attrs.GetAttr<int32_t>("groups"));
  ctx->offset_groups = JUST(composed_attrs.GetAttr<int32_t>("offset_groups"));

  return Maybe<void>::Ok();
}

Maybe<void> DeformConvNd::Apply(const DeformConvNdCaptureState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  in_grads->resize(5);
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& input = ctx->SavedTensors().at(0);
  const auto& weight = ctx->SavedTensors().at(1);
  const auto& offset = ctx->SavedTensors().at(2);
  const auto& mask = ctx->SavedTensors().at(3);
  const auto& output_grad = out_grads.at(0);
  if (ctx->input_requires_grad || ctx->offset_requires_grad || ctx->mask_requires_grad) {
    std::shared_ptr<TensorTuple> grads_tuple;
    if (ctx->use_mask) {
      grads_tuple = JUST(functional::DeformConv2dInputGrad(
          output_grad, input, weight, offset, mask, ctx->stride_h, ctx->stride_w, ctx->pad_h,
          ctx->pad_w, ctx->dilation_h, ctx->dilation_w, ctx->groups, ctx->offset_groups,
          ctx->use_mask));
    } else {
      grads_tuple = JUST(functional::DeformConv2dInputGrad(
          output_grad, input, weight, offset, NullOpt, ctx->stride_h, ctx->stride_w, ctx->pad_h,
          ctx->pad_w, ctx->dilation_h, ctx->dilation_w, ctx->groups, ctx->offset_groups,
          ctx->use_mask));
    }
    if (ctx->input_requires_grad) {
      in_grads->at(0) = grads_tuple->at(0);  // input_grad
    }
    if (ctx->offset_requires_grad) {
      in_grads->at(2) = grads_tuple->at(1);  // offset_grad
    }
    if (ctx->use_mask && ctx->mask_requires_grad) {
      in_grads->at(3) = grads_tuple->at(2);  // mask_grad
    }
  }

  if (ctx->weight_requires_grad) {  // weight_grad
    in_grads->at(1) = JUST(functional::DeformConv2dParamGrad(
        output_grad, input, weight, offset, mask, ctx->stride_h, ctx->stride_w, ctx->pad_h,
        ctx->pad_w, ctx->dilation_h, ctx->dilation_w, ctx->groups, ctx->offset_groups,
        ctx->use_mask));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("deform_conv2d", DeformConvNd);

}  // namespace one
}  // namespace oneflow