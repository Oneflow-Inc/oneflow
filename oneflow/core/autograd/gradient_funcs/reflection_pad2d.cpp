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

struct ReflectionPad2dInterpState : public OpExprInterpState {
  bool requires_grad;
};

class ReflectionPad2d : public OpExprGradFunction<ReflectionPad2dInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ReflectionPad2dInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ReflectionPad2dInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  std::shared_ptr<std::vector<int64_t>> padding_;
  double floating_value_;
  int64_t integral_value_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> ReflectionPad2d::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());
  padding_ = JUST(op_trait_->GetAttr<std::vector<int64_t>>("padding"));
  floating_value_ = JUST(op_trait_->GetAttr<double>("floating_value"));
  integral_value_ = JUST(op_trait_->GetAttr<int64_t>("integral_value"));
  grad_op_ = JUST(op_expr_helper::ReflectionPad2dGradOp(*padding_, floating_value_, integral_value_,
                                                        GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> ReflectionPad2d::Capture(ReflectionPad2dInterpState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReflectionPad2d::Apply(const ReflectionPad2dInterpState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const auto& dy = out_grads.at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {dy}, {}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reflection_pad2d", ReflectionPad2d);

}  // namespace one
}  // namespace oneflow
