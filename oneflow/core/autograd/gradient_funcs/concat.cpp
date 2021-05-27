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
#include "oneflow/core/framework/user_op_conf_trait.h"

namespace oneflow {
namespace one {

struct ConcatInterpState : public OpExprInterpState {
  bool requires_grad;
  int64_t axis;
};

class Concat : public OpExprGradFunction<ConcatInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ConcatInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const ConcatInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> Concat::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());
  int32_t input_num = JUST(op_trait_->input_size("in"));
  int64_t axis;
  grad_op_ = JUST(op_expr_helper::SplitLikeOp(input_num, axis, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> Concat::Capture(ConcatInterpState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  int input_len = inputs.size();
  for (int i = 0; i < input_len; i++) {
    ctx->requires_grad = ctx->requires_grad | inputs.at(i)->requires_grad();
    if (ctx->requires_grad == true) break;
  }
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("axis"));
  for (int i = 0; i < input_len; i++) { ctx->SaveTensorForBackward(inputs.at(i)); }
  return Maybe<void>::Ok();
}

Maybe<void> Concat::Apply(const ConcatInterpState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const int n = (*in_grads).size();
  TensorTuple like;
  like.reserve((*in_grads).size() + 1);
  like.push_back(out_grads.at(0));
  for (int i = 0; i < n; i++) { like.push_back(ctx->SavedTensors().at(i)); }
  MutableAttrMap concat_attrs;
  int64_t axis = ctx->axis;
  JUST(concat_attrs.SetAttr<int64_t>("axis", axis));
  const auto& results = JUST(OpInterpUtil::Dispatch<TensorTuple>(*grad_op_, like, concat_attrs));
  CHECK_EQ_OR_RETURN(results->size(), n);
  for (int i = 0; i < n; i++) { in_grads->at(i) = results->at(i); }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("concat", Concat);

}  // namespace one
}  // namespace oneflow
