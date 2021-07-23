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

struct WhereInterpState : public OpExprInterpState {
  bool requires_grad_x;
  bool requires_grad_y;
};

class Where : public OpExprGradFunction<WhereInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(WhereInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const WhereInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> zero_like_op_;
  std::shared_ptr<OpExpr> where_op_x_;
  std::shared_ptr<OpExpr> where_op_y_;
};

Maybe<void> Where::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  zero_like_op_ = JUST(op_expr_helper::ZeroLikeOp("zeros_like_" + GradientOpName(op_name)));
  where_op_x_ = JUST(op_expr_helper::WhereOp("where_x_" + GradientOpName(op_name)));
  where_op_y_ = JUST(op_expr_helper::WhereOp("where_y_" + GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> Where::Capture(WhereInterpState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_x = inputs.at(1)->requires_grad();
  ctx->requires_grad_y = inputs.at(2)->requires_grad();
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->SaveTensorForBackward(inputs.at(0));  // condition
  ctx->SaveTensorForBackward(inputs.at(1));  // x
  return Maybe<void>::Ok();
}

Maybe<void> Where::Apply(const WhereInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  MutableAttrMap attrs;
  const std::shared_ptr<oneflow::one::Tensor>& condtion = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(1);

  std::shared_ptr<oneflow::one::Tensor> zero_out =
      JUST(OpInterpUtil::Dispatch<Tensor>(*zero_like_op_, {x}));
  in_grads->resize(3);
  if (ctx->requires_grad_x)
    in_grads->at(1) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*where_op_x_, {condtion, out_grads.at(0), zero_out}));
  if (ctx->requires_grad_y)
    in_grads->at(2) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*where_op_y_, {condtion, zero_out, out_grads.at(0)}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("where", Where);

}  // namespace one
}  // namespace oneflow
