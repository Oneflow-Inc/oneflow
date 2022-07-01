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

struct AdaptivePoolCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class AdaptivePoolNdGrad : public OpExprGradFunction<AdaptivePoolCaptureState> {
 public:
  using OpExprGradFunction<AdaptivePoolCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, std::string mode, const int& ndims);
  Maybe<void> Capture(AdaptivePoolCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const AdaptivePoolCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::string mode_;
  int32_t ndims_;
};

Maybe<void> AdaptivePoolNdGrad::Init(const OpExpr& op, std::string mode, const int& ndims) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  mode_ = mode;
  ndims_ = ndims;
  return Maybe<void>::Ok();
}

Maybe<void> AdaptivePoolNdGrad::Capture(AdaptivePoolCaptureState* ctx, const TensorTuple& inputs,
                                        const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> AdaptivePoolNdGrad::Apply(const AdaptivePoolCaptureState* ctx,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::AdaptivePoolNdGrad(x, out_grads.at(0), mode_, ndims_));
  return Maybe<void>::Ok();
}

class AdaptiveAvgPool1dGrad final : public AdaptivePoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptivePoolNdGrad::Init(op, "avg", 1); }
};

class AdaptiveAvgPool2dGrad final : public AdaptivePoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptivePoolNdGrad::Init(op, "avg", 2); }
};

class AdaptiveAvgPool3dGrad final : public AdaptivePoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptivePoolNdGrad::Init(op, "avg", 3); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool1d", AdaptiveAvgPool1dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool2d", AdaptiveAvgPool2dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool3d", AdaptiveAvgPool3dGrad);

}  // namespace one
}  // namespace oneflow
