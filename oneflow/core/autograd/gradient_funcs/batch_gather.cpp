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
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct BatchGatherInterpState : public OpExprInterpState {
  int64_t num_segments;
  bool requires_grad;
};

class BatchGather : public OpExprGradFunction<BatchGatherInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BatchGatherInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BatchGatherInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> bw_unsorted_batch_segment_sum_op_;
};

Maybe<void> BatchGather::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  bw_unsorted_batch_segment_sum_op_ =
      JUST(op_expr_helper::UnsortedBatchSegmentSumOp(/*num_segments=*/1, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Capture(BatchGatherInterpState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& in_shape = inputs.at(0)->shape();
  const auto& indices_shape = inputs.at(1)->shape();
  ctx->num_segments = in_shape->At(indices_shape->NumAxes() - 1);
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Apply(const BatchGatherInterpState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& indices = ctx->SavedTensors().at(0);
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<int32_t>("num_segments", ctx->num_segments));
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*bw_unsorted_batch_segment_sum_op_,
                                                        {out_grads.at(0), indices}, attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("batch_gather", BatchGather);

}  // namespace one
}  // namespace oneflow
