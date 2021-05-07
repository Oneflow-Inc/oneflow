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
#include "oneflow/core/framework/attr_map_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct ReduceSumOpInterpState : public OpExprInterpState {
  std::vector<int32_t> axis;
};

class ReduceSumOp : public OpExprGradFunction<ReduceSumOpInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReduceSumOpInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReduceSumOpInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> ReduceSumOp::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMap(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  grad_op_ = JUST(op_expr_helper::BroadcastLikeOp(/*axis=*/{-1}, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSumOp::Capture(ReduceSumOpInterpState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSumOp::Apply(const ReduceSumOpInterpState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  const auto& input = ctx->SavedTensors().at(0);
  const auto& dy = out_grads.at(0);
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<std::vector<int32_t>>("axis", ctx->axis));
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {dy, input}, attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_sum", ReduceSumOp);

struct ReduceMaxOrMinOpInterpState : public OpExprInterpState {
  std::vector<int32_t> axis;
  bool keepdims;
};

class ReduceMaxOrMinOp : public OpExprGradFunction<ReduceMaxOrMinOpInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReduceMaxOrMinOpInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReduceMaxOrMinOpInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> bcast_like_op_;
  std::shared_ptr<OpExpr> bcast_equal_op_;
  std::shared_ptr<OpExpr> cast_like_op_;
  std::shared_ptr<OpExpr> reduce_sum_op_;
  std::shared_ptr<OpExpr> bcast_div_op_;
  std::shared_ptr<OpExpr> multiply_op_;
};

Maybe<void> ReduceMaxOrMinOp::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMap(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  bcast_like_op_ =
      JUST(op_expr_helper::BroadcastLikeOp(/*axis=*/{-1}, GradientOpName(op_name + "_bcast_like")));
  bcast_equal_op_ = JUST(op_expr_helper::BroadcastEqualOp(GradientOpName(op_name + "_bcast_eq")));
  cast_like_op_ = JUST(op_expr_helper::CastLikeOp(GradientOpName(op_name + "_cast_like")));
  reduce_sum_op_ = JUST(op_expr_helper::ReduceSumOp(/*axis=*/{-1}, /*keepdims=*/false,
                                                    GradientOpName(op_name + "_reduce_sum")));
  bcast_div_op_ = JUST(op_expr_helper::BroadcastDivOp(GradientOpName(op_name + "_bcast_div")));
  multiply_op_ = JUST(op_expr_helper::MultiplyOp(op_name + "_multiply"));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceMaxOrMinOp::Capture(ReduceMaxOrMinOpInterpState* ctx, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceMaxOrMinOp::Apply(const ReduceMaxOrMinOpInterpState* ctx,
                                    const TensorTuple& out_grads, TensorTuple* in_grads) const {
  const auto& input = ctx->SavedTensors().at(0);
  const auto& output = ctx->SavedTensors().at(1);
  const auto& dy = out_grads.at(0);

  MutableAttrMap bcast_attrs;
  JUST(bcast_attrs.SetAttr<std::vector<int32_t>>("axis", ctx->axis));
  const auto& bcast_like =
      JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_like_op_, {output, input}, bcast_attrs));
  const auto& bcast_eq =
      JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_equal_op_, {input, bcast_like}));
  const auto& cast_like = JUST(OpInterpUtil::Dispatch<Tensor>(*cast_like_op_, {bcast_eq, input}));

  MutableAttrMap reduce_sum_attrs;
  JUST(reduce_sum_attrs.SetAttr<std::vector<int32_t>>("axis", ctx->axis));
  JUST(reduce_sum_attrs.SetAttr<bool>("keepdims", ctx->keepdims));
  const auto& reduce_sum =
      JUST(OpInterpUtil::Dispatch<Tensor>(*reduce_sum_op_, {cast_like}, reduce_sum_attrs));
  const auto& broadcast_div =
      JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_div_op_, {dy, reduce_sum}));
  const auto& bcast_like_div =
      JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_like_op_, {broadcast_div, input}, bcast_attrs));

  in_grads->resize(1);
  in_grads->at(0) =
      JUST(OpInterpUtil::Dispatch<Tensor>(*multiply_op_, {bcast_like_div, cast_like}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min", ReduceMaxOrMinOp);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max", ReduceMaxOrMinOp);

}  // namespace one
}  // namespace oneflow
