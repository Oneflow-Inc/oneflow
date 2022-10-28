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

struct PartialFCSampleState : public AutoGradCaptureState {
  bool requires_grad = false;
  int32_t index_sampled_label = -1;
  int32_t index_weight = -1;
};

class PartialFCSample : public OpExprGradFunction<PartialFCSampleState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(PartialFCSampleState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const PartialFCSampleState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> PartialFCSample::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> PartialFCSample::Capture(PartialFCSampleState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->index_sampled_label = ctx->SaveTensorForBackward(outputs.at(1));  // sampled_label
  ctx->index_weight = ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> PartialFCSample::Apply(const PartialFCSampleState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 3);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(2);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& diff_sampled_weight = out_grads.at(2);  // diff of sampled_weight

  const auto& sampled_tensor = ctx->SavedTensors().at(ctx->index_sampled_label);
  const auto& weight = ctx->SavedTensors().at(ctx->index_weight);
  const auto& out_tensors_of_op0 = JUST(
      functional::DistributedPariticalFCSampleDisableBoxing(diff_sampled_weight, sampled_tensor));

  const auto& out_tensors_of_op1 = JUST(functional::UnsortedSegmentSumLike(
      out_tensors_of_op0->at(0), out_tensors_of_op0->at(1), weight, 0));
  in_grads->at(0) = out_tensors_of_op1;
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("distributed_partial_fc_sample", PartialFCSample);

}  // namespace one
}  // namespace oneflow
