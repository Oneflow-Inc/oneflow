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

struct DropoutCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  bool has_addend = false;
  float rate = 0.0;
};

class Dropout : public OpExprGradFunction<DropoutCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DropoutCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DropoutCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Dropout::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Dropout::Capture(DropoutCaptureState* ctx, const TensorTuple& inputs,
                             const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->requires_grad = inputs.at(0)->requires_grad();

  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->rate = JUST(composed_attrs.GetAttr<float>("rate"));

  if (inputs.size() == 1) {
    ctx->has_addend = false;
  } else if (inputs.size() == 2) {
    ctx->has_addend = true;
  } else {
    UNIMPLEMENTED();
  }

  ctx->SaveTensorForBackward(outputs.at(1));  // output mask
  return Maybe<void>::Ok();
}

Maybe<void> Dropout::Apply(const DropoutCaptureState* ctx, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // Output has y and mask.
  float scale = 0.0f;                       // When dropout rate = 1.0, we set scale as zero.
  if (ctx->rate < 1.0f) { scale = 1.0f / (1.0f - ctx->rate); }
  const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
  if (ctx->has_addend) {
    in_grads->resize(2);
    in_grads->at(0) = JUST(functional::DropoutGrad(out_grads.at(0), mask, scale));
    in_grads->at(1) = out_grads.at(0);
    return Maybe<void>::Ok();
  } else {
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::DropoutGrad(out_grads.at(0), mask, scale));
    return Maybe<void>::Ok();
  }
}

REGISTER_OP_EXPR_GRAD_FUNCTION("dropout", Dropout);

}  // namespace one
}  // namespace oneflow
