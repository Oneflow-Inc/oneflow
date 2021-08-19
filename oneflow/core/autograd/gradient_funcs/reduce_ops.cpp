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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ReduceSumCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
};

class ReduceSum : public OpExprGradFunction<ReduceSumCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReduceSumCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReduceSumCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ReduceSum::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSum::Capture(ReduceSumCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSum::Apply(const ReduceSumCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& input = ctx->SavedTensors().at(0);
  const auto& dy = out_grads.at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::BroadcastLike(dy, input, ctx->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_sum", ReduceSum);

struct ReduceMaxOrMinCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool keepdims;
};

class ReduceMaxOrMin : public OpExprGradFunction<ReduceMaxOrMinCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReduceMaxOrMinCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReduceMaxOrMinCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ReduceMaxOrMin::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ReduceMaxOrMin::Capture(ReduceMaxOrMinCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
  ctx->keepdims = JUST(composed_attrs.GetAttr<bool>("keepdims"));
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceMaxOrMin::Apply(const ReduceMaxOrMinCaptureState* ctx,
                                  const TensorTuple& out_grads, TensorTuple* in_grads) const {
  const auto& input = ctx->SavedTensors().at(0);
  const auto& output = ctx->SavedTensors().at(1);
  const auto& dy = out_grads.at(0);

  const auto& bcast_like = JUST(functional::BroadcastLike(output, input, ctx->axis));
  const auto& bcast_eq = JUST(functional::BroadcastEqual(input, bcast_like));
  const auto& cast_like = JUST(functional::CastLike(bcast_eq, input));
  const auto& reduce_sum_ = JUST(functional::ReduceSum(cast_like, ctx->axis, ctx->keepdims));
  const auto& bcast_div_ = JUST(functional::BroadcastDiv(dy, reduce_sum_));
  const auto& bcast_like_div = JUST(functional::BroadcastLike(bcast_div_, input, ctx->axis));

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Multiply(bcast_like_div, cast_like));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min", ReduceMaxOrMin);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max", ReduceMaxOrMin);

}  // namespace one
}  // namespace oneflow
