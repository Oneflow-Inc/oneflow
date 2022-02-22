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

struct FusedScaleTrilSoftmaxMaskScaleInterpState : public AutoGradCaptureState {
  bool input_requires_grad = true;
  int64_t diagonal = 0;
  float tril_scale_value = 0.0;
  float mask_scale_value = 1.0;
};

class FusedScaleTrilSoftmaxMaskScale
    : public OpExprGradFunction<FusedScaleTrilSoftmaxMaskScaleInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedScaleTrilSoftmaxMaskScaleInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedScaleTrilSoftmaxMaskScaleInterpState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedScaleTrilSoftmaxMaskScale::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleTrilSoftmaxMaskScale::Capture(FusedScaleTrilSoftmaxMaskScaleInterpState* ctx,
                                                    const TensorTuple& inputs,
                                                    const TensorTuple& outputs,
                                                    const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  ctx->input_requires_grad = inputs.at(0)->requires_grad();  // input

  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->diagonal = JUST(composed_attrs.GetAttr<int64_t>("diagonal"));
  ctx->tril_scale_value = JUST(composed_attrs.GetAttr<float>("tril_scale_value"));
  ctx->mask_scale_value = JUST(composed_attrs.GetAttr<float>("mask_scale_value"));
  ctx->SaveTensorForBackward(inputs.at(1));   // Save Mask
  ctx->SaveTensorForBackward(outputs.at(1));  // Save softmax_y
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleTrilSoftmaxMaskScale::Apply(
    const FusedScaleTrilSoftmaxMaskScaleInterpState* ctx, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // Cause output has y and softmax_y
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  // mask have no grad(reqiures_grad=False), but still take a place in in_grads
  in_grads->resize(2);

  const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& softmax_y = ctx->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& input_grad =
      JUST(functional::FusedScaleTrilSoftmaxMaskScaleGrad(softmax_y, out_grads.at(0), mask,
                                                          ctx->diagonal, ctx->tril_scale_value,
                                                          ctx->mask_scale_value));
  if (ctx->input_requires_grad) { in_grads->at(0) = input_grad; }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_tril_scale_softmax_mask_scale",
                               FusedScaleTrilSoftmaxMaskScale);

}  // namespace one
}  // namespace oneflow
