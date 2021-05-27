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
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> Concat::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  int64_t axis;
  grad_op_ = JUST(op_expr_helper::ConcatGradOp(axis, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> Concat::Capture(ConcatInterpState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  int input_len = inputs.size();
  for (int i = 0; i < input_len; i++) {
    ctx->requires_grad = ctx->requires_grad | inputs.at(i)->requires_grad();
  }
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  for (int i = 0; i < input_len; i++) { ctx->SaveTensorForBackward(inputs.at(i)); }
  ctx->axis = JUST(composed_attrs.GetAttr<int64_t>("axis"));
  return Maybe<void>::Ok();
}

Maybe<void> Concat::Apply(const ConcatInterpState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  MutableAttrMap attrs;
  JUST(attrs.SetAttr<int64_t>("axis", ctx->axis));
  int input_len = (*in_grads).size();
  in_grads->resize(input_len);
  for (int i = 0; i < input_len; i++) {
    const std::shared_ptr<oneflow::one::Tensor>& like = ctx->SavedTensors().at(i);
    in_grads->at(i) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0), like}, attrs));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("concat", Concat);

}  // namespace one
}  // namespace oneflow
