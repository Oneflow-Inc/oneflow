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

struct AdaptiveMaxPoolCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
};

class AdaptiveMaxPoolNdGrad : public OpExprGradFunction<AdaptiveMaxPoolCaptureState> {
 public:
  using OpExprGradFunction<AdaptiveMaxPoolCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, const int& ndims);
  Maybe<void> Capture(AdaptiveMaxPoolCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const AdaptiveMaxPoolCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  int32_t ndims_ = 0;
};

Maybe<void> AdaptiveMaxPoolNdGrad::Init(const OpExpr& op, const int& ndims) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  ndims_ = ndims;
  return Maybe<void>::Ok();
}

Maybe<void> AdaptiveMaxPoolNdGrad::Capture(AdaptiveMaxPoolCaptureState* ctx,
                                           const TensorTuple& inputs, const TensorTuple& outputs,
                                           const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(outputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> AdaptiveMaxPoolNdGrad::Apply(const AdaptiveMaxPoolCaptureState* ctx,
                                         const TensorTuple& out_grads,
                                         TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& index = ctx->SavedTensors().at(1);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::AdaptiveMaxPoolNdGrad(x, out_grads.at(0), index, ndims_));
  return Maybe<void>::Ok();
}

class AdaptiveMaxPool1dGrad final : public AdaptiveMaxPoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptiveMaxPoolNdGrad::Init(op, 1); }
};

class AdaptiveMaxPool2dGrad final : public AdaptiveMaxPoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptiveMaxPoolNdGrad::Init(op, 2); }
};

class AdaptiveMaxPool3dGrad final : public AdaptiveMaxPoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return AdaptiveMaxPoolNdGrad::Init(op, 3); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool1d", AdaptiveMaxPool1dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool2d", AdaptiveMaxPool2dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool3d", AdaptiveMaxPool3dGrad);

}  // namespace one
}  // namespace oneflow
