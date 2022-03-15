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

struct FusedScaleMaskSoftmaxInterState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  float scale = 1.0;
};

class FusedScaleMaskSoftmax : public OpExprGradFunction<FusedScaleMaskSoftmaxInterState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedScaleMaskSoftmaxInterState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedScaleMaskSoftmaxInterState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedScaleMaskSoftmax::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmax::Capture(FusedScaleMaskSoftmaxInterState* ctx,
                                           const TensorTuple& inputs, const TensorTuple& outputs,
                                           const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);  // input, mask
  ctx->input_requires_grad = inputs.at(0)->requires_grad();

  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->scale = JUST(composed_attrs.GetAttr<float>("scale_value"));

  ctx->SaveTensorForBackward(inputs.at(1));   // save mask
  ctx->SaveTensorForBackward(outputs.at(0));  // save y, ie. softmax result
  return Maybe<void>::Ok();
}

Maybe<void> FusedScaleMaskSoftmax::Apply(const FusedScaleMaskSoftmaxInterState* ctx,
                                         const TensorTuple& out_grads,
                                         TensorTuple* in_grads) const {
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // dy
  in_grads->resize(2);                      // input, mask

  const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(1);
  const std::shared_ptr<oneflow::one::Tensor>& fused_scale_mask_softmax_grad =
      JUST(functional::FusedScaleMaskSoftmaxGrad(y, out_grads.at(0), mask, ctx->scale));

  in_grads->at(0) = fused_scale_mask_softmax_grad;
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_scale_mask_softmax", FusedScaleMaskSoftmax);

}  // namespace one
}  // namespace oneflow
