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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BatchGatherCaptureState : public AutoGradCaptureState {
  int64_t num_segments;
  bool requires_grad;
};

class BatchGather : public OpExprGradFunction<BatchGatherCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BatchGatherCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BatchGatherCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BatchGather::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Capture(BatchGatherCaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& in_shape = inputs.at(0)->shape();
  const auto& indices_shape = inputs.at(1)->shape();
  ctx->num_segments = in_shape->At(indices_shape->NumAxes() - 1);
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Apply(const BatchGatherCaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& indices = ctx->SavedTensors().at(0);
  in_grads->at(0) =
      JUST(functional::UnsortedBatchSegmentSum(out_grads.at(0), indices, ctx->num_segments));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("batch_gather", BatchGather);

}  // namespace one
}  // namespace oneflow
