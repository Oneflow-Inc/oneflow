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
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct WkvGradCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  int64_t B;
  int64_t T;
  int64_t C;
};

class WkvGrad : public OpExprGradFunction<WkvGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(WkvGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const WkvGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> WkvGrad::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> WkvGrad::Capture(WkvGradCaptureState* ctx, const TensorTuple& inputs,
                             const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs[0]->requires_grad();
  ctx->B = JUST(attrs.GetAttr<int64_t>("B"));
  ctx->T = JUST(attrs.GetAttr<int64_t>("T"));
  ctx->C = JUST(attrs.GetAttr<int64_t>("C"));
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));
  ctx->SaveTensorForBackward(inputs.at(2));
  ctx->SaveTensorForBackward(inputs.at(3));
  return Maybe<void>::Ok();
}

Maybe<void> WkvGrad::Apply(const WkvGradCaptureState* ctx, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "out_grads.size() must be equal to 1.";
  in_grads->resize(4);
  if (ctx->requires_grad) {
    const std::shared_ptr<oneflow::one::Tensor>& w = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& u = ctx->SavedTensors().at(1);
    const std::shared_ptr<oneflow::one::Tensor>& k = ctx->SavedTensors().at(2);
    const std::shared_ptr<oneflow::one::Tensor>& v = ctx->SavedTensors().at(3);

    const auto& outputs =
        JUST(functional::WkvGrad(ctx->B, ctx->T, ctx->C, w, u, k, v, out_grads[0]));
    (*in_grads)[0] = (*outputs)[0];
    (*in_grads)[1] = (*outputs)[1];
    (*in_grads)[2] = (*outputs)[2];
    (*in_grads)[3] = (*outputs)[3];
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("wkv", WkvGrad);

}  // namespace one
}  // namespace oneflow
