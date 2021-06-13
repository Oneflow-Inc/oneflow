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

struct AdaptivePoolInterpState : public OpExprInterpState {
  bool requires_grad;
};

class AdaptivePool : public OpExprGradFunction<AdaptivePoolInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(AdaptivePoolInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const AdaptivePoolInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> AdaptivePool::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  grad_op_ = JUST(op_expr_helper::AdaptivePoolGradOp(GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> AdaptivePool::Capture(AdaptivePoolInterpState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> AdaptivePool::Apply(const AdaptivePoolInterpState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {x, out_grads.at(0)}, {}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool2d", AdaptivePool);

}  // namespace one
}  // namespace oneflow
