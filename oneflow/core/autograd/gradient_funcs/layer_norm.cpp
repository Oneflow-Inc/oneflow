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

// y, mean, inv_variance, [normalized] =
//   layer_norm(x, [beta], [gamma], center=False, scale=False, begin_norm_axis=1,
//              begin_params_axis=-1, epsilon=1e-5)
class LayerNorm : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  bool center_;
  bool scale_;
  int64_t begin_norm_axis_;
  int64_t begin_params_axis_;
  double epsilon_;
  mutable bool has_beta_diff_;
  mutable bool has_gamma_diff_;
  mutable bool has_normalized_diff_;
  mutable bool x_requires_grad_;
};

Maybe<void> LayerNorm::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  center_ = GetAttr<bool>(fw_op_expr->proto(), "center");
  scale_ = GetAttr<bool>(fw_op_expr->proto(), "scale");
  begin_norm_axis_ = GetAttr<int64_t>(fw_op_expr->proto(), "begin_norm_axis");
  begin_params_axis_ = GetAttr<int64_t>(fw_op_expr->proto(), "begin_params_axis");
  epsilon_ = GetAttr<double>(fw_op_expr->proto(), "epsilon");
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), center_ + scale_ + 1);
  CHECK_EQ_OR_RETURN(inputs.size(), scale_ + 3);
  has_beta_diff_ = center_ && inputs.at(1)->requires_grad();
  const int gamma_index = center_ + 1;
  has_gamma_diff_ = scale_ && inputs.at(gamma_index)->requires_grad();
  has_normalized_diff_ = scale_ && inputs.at(0)->requires_grad();
  if (has_gamma_diff_ || has_normalized_diff_) {
    ctx->SaveTensorForBackward(inputs.at(gamma_index));
  }
  if (has_gamma_diff_) { ctx->SaveTensorForBackward(outputs.at(3)); }
  x_requires_grad_ = inputs.at(0)->requires_grad();
  if (x_requires_grad_) {
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(outputs.at(0));
    ctx->SaveTensorForBackward(outputs.at(1));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  in_grads->resize(center_ + scale_ + 1);
  const auto& saved_tensors = ctx->SavedTensors();
  std::shared_ptr<Tensor> dy = out_grads.at(0);
  int64_t begin_params_axis = begin_params_axis_;
  if (begin_params_axis < 0) { begin_params_axis += dy->shape()->NumAxes(); }
  int64_t begin_norm_axis = begin_norm_axis_;
  if (begin_norm_axis < 0) { begin_norm_axis += dy->shape()->NumAxes(); }
  int offset = 0;
  if (has_beta_diff_ || has_gamma_diff_ || has_normalized_diff_) {
    const auto& param_grad_op = JUST(op_expr_helper::LayerNormParamGradOp(
        begin_params_axis, has_beta_diff_, has_gamma_diff_, has_normalized_diff_,
        GradientOpName(op_name_ + "_param")));
    TensorTuple inputs{dy};
    if (has_gamma_diff_ || has_normalized_diff_) {
      inputs.push_back(saved_tensors.at(offset++));  // gamma
    }
    if (has_gamma_diff_) {
      inputs.push_back(saved_tensors.at(offset++));  // normalized
    }
    const auto& results = JUST(Dispatch<TensorTuple>(*param_grad_op, inputs));
    if (has_beta_diff_) { in_grads->at(1) = results->at(0); }
    if (has_gamma_diff_) { in_grads->at(has_beta_diff_ + 1) = results->at(has_beta_diff_); }
    if (has_normalized_diff_) { dy = results->at(has_beta_diff_ + has_gamma_diff_); }
  }
  if (x_requires_grad_) {
    const auto& grad_op =
        JUST(op_expr_helper::LayerNormGradOp(begin_norm_axis, epsilon_, GradientOpName(op_name_)));
    CHECK_EQ_OR_RETURN(saved_tensors.size(), offset + 3);
    const auto& x = saved_tensors.at(offset);
    const auto& mean = saved_tensors.at(offset + 1);
    const auto& inv_variance = saved_tensors.at(offset + 2);
    in_grads->at(0) = JUST(Dispatch<Tensor>(*grad_op, {x, mean, inv_variance, dy}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("layer_norm", LayerNorm);

}  // namespace one
}  // namespace oneflow
