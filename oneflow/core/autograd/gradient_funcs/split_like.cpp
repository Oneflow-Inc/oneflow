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
#include "oneflow/core/autograd/gradient_funcs/utility.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_dispatch.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

class SplitLike : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  int64_t axis_;
  mutable int64_t max_dim_size_;
  mutable bool requires_grad_;
};

Maybe<void> SplitLike::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  axis_ = GetAttr<int64_t>(fw_op_expr->proto(), "axis");
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), outputs.size() + 1);
  requires_grad_ = inputs.at(0)->requires_grad();
  if (!requires_grad_) { return Maybe<void>::Ok(); }
  max_dim_size_ = 0;
  for (int i = 0; i < outputs.size(); ++i) {
    max_dim_size_ += inputs.at(i + 1)->shape()->At(axis_);
    ctx->SaveTensorForBackward(outputs.at(i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  in_grads->resize(1);
  if (!requires_grad_) { return Maybe<void>::Ok(); }

  const auto& saved_tensors = ctx->SavedTensors();
  TensorTuple inputs;
  for (int i = 0; i < out_grads.size(); ++i) {
    const auto& out_grad_i = out_grads.at(i);
    if (out_grad_i.get()) {
      inputs.push_back(out_grad_i);
    } else {
      const auto& zero_like_op = JUST(
          op_expr_helper::ZeroLikeOp(GradientOpName(op_name_ + "_zero_grad_" + std::to_string(i))));
      const auto& zero_grad = JUST(Dispatch<Tensor>(*zero_like_op, {saved_tensors.at(i)}));
      inputs.push_back(zero_grad);
    }
  }
  const auto& grad_op = JUST(op_expr_helper::ConcatOp(out_grads.size(), axis_, max_dim_size_));
  in_grads->at(0) = JUST(Dispatch<Tensor>(*grad_op, inputs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("split_like", SplitLike);

}  // namespace one
}  // namespace oneflow
