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
#include "oneflow/core/functional/sequence_function.h"

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
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
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
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_nansum", ReduceSum);

struct ReduceProdOpInterpState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad;
};

class ReduceProdOp : public OpExprGradFunction<ReduceProdOpInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReduceProdOpInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReduceProdOpInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ReduceProdOp::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ReduceProdOp::Capture(ReduceProdOpInterpState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
  ctx->requires_grad = inputs.at(0)->requires_grad();
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceProdOp::Apply(const ReduceProdOpInterpState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  const auto& input = ctx->SavedTensors().at(0);
  const auto& output = ctx->SavedTensors().at(1);
  const auto& dy = out_grads.at(0);

  in_grads->resize(1);
  in_grads->at(0) = JUST(
      functional::SequenceFunction<Maybe<Tensor>()>([&]() { return functional::Mul(dy, output); })
          .then(std::bind(functional::BroadcastLike, std::placeholders::_1, input, ctx->axis))
          .then(std::bind(functional::Div, std::placeholders::_1, input))
          .call());
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_prod", ReduceProdOp);

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
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
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

  const auto cast_like =
      JUST(functional::SequenceFunction<Maybe<Tensor>()>(
               [&]() { return functional::BroadcastLike(output, input, ctx->axis); })
               .then(std::bind(functional::BroadcastEqual, input, std::placeholders::_1))
               .then(std::bind(functional::CastLike, std::placeholders::_1, input))
               .call());

  const auto& bcast_like_div =
      JUST(functional::SequenceFunction<Maybe<Tensor>()>(
               [&]() { return functional::ReduceSum(cast_like, ctx->axis, ctx->keepdims); })
               .then(std::bind(functional::Div, dy, std::placeholders::_1))
               .then(std::bind(functional::BroadcastLike, std::placeholders::_1, input, ctx->axis))
               .call());

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Mul(bcast_like_div, cast_like));

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min", ReduceMaxOrMin);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max", ReduceMaxOrMin);

}  // namespace one
}  // namespace oneflow
