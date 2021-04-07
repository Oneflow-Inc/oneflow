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

class BatchGather : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  mutable int64_t num_segments_;
};

Maybe<void> BatchGather::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs) const {
  requires_grad_ = inputs.at(0)->requires_grad();
  if (!requires_grad_) { return Maybe<void>::Ok(); }
  const auto& in_shape = inputs.at(0)->shape();
  const auto& indices_shape = inputs.at(1)->shape();
  num_segments_ = in_shape->At(indices_shape->NumAxes() - 1);
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> BatchGather::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (!requires_grad_) { return Maybe<void>::Ok(); }
  const auto& indices = ctx->SavedTensors().at(0);
  const auto& grad_op =
      JUST(op_expr_helper::UnsortedBatchSegmentSumOp(num_segments_, GradientOpName(op_name_)));
  in_grads->at(0) = JUST(Dispatch<Tensor>(*grad_op, {out_grads.at(0), indices}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("batch_gather", BatchGather);

}  // namespace one
}  // namespace oneflow
