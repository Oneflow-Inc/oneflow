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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_dispatch.h"
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {
namespace one {

// training:
// y, mean, inv_variance = normalization(x, moving_mean, moving_variance, gamma, beta,
// axis=1, epsilon=0.01, momentum=0.9)
// inference:
// y = normalization(x, moving_mean, moving_variance, gamma, beta, axis=1, epsilon=0.01,
// momentum=0.9)
class NormalizationGrad : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    op_name_ = fw_op_expr->op_name();
    const float epsilon = GetAttr<float>(fw_op_expr->proto(), "epsilon");
    axis_ = GetAttr<int32_t>(fw_op_expr->proto(), "axis");
    is_training_ = GetAttr<bool>(fw_op_expr->proto(), "training");
    // v1 = variance + eps
    add_eps_op_ = JUST(op_expr_helper::ScalarAddOp(epsilon));
    // v2 = rsqrt(v1)
    rsqrt_op_ = JUST(op_expr_helper::RsqrtOp());
    // Normalization grad.
    normalization_grad_op_ = JUST(op_expr_helper::NormalizationGradOp(axis_, epsilon));

    broadcast_mul_op_ = JUST(op_expr_helper::BroadcastMulOp());
    // fp16 -> fp32 and fp32 -> fp16.
    h2f_cast_op_ = JUST(op_expr_helper::CastOp(DataType::kFloat));
    f2h_cast_op_ = JUST(op_expr_helper::CastOp(DataType::kFloat16));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    ctx->SaveTensorForBackward(inputs.at(0));  // x
    ctx->SaveTensorForBackward(inputs.at(3));  // gamma
    if (is_training_) {
      ctx->SaveTensorForBackward(outputs.at(1));  // mean
      ctx->SaveTensorForBackward(outputs.at(2));  // inv_variance
    } else {
      ctx->SaveTensorForBackward(inputs.at(1));  // moving_mean
      ctx->SaveTensorForBackward(inputs.at(2));  // moving_variance
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);      // x
    const auto& gamma = ctx->SavedTensors().at(1);  // gamma
    const auto& y_grad = out_grads.at(0);

    std::shared_ptr<Tensor> mean, inv_variance;
    if (is_training_) {
      mean = ctx->SavedTensors().at(2);          // mean
      inv_variance = ctx->SavedTensors().at(3);  // inv_variance
    } else {
      const auto& moving_mean = ctx->SavedTensors().at(2);      // moving_mean
      const auto& moving_variance = ctx->SavedTensors().at(3);  // moving_variance
      const auto& add_eps = JUST(Dispatch<Tensor>(*add_eps_op_, {moving_variance}));
      mean = moving_mean;
      inv_variance = JUST(Dispatch<Tensor>(*rsqrt_op_, {add_eps}));
    }
    const auto& results = JUST(
        Dispatch<TensorTuple>(*normalization_grad_op_, {x, y_grad, gamma, mean, inv_variance}));
    CHECK_EQ_OR_RETURN(results->size(), 3);
    // The normalization op has 5 inputs which are x, moving_mean, moving_variance, gamma and beta.
    in_grads->resize(5);
    in_grads->at(3) = results->at(1);  // gamma_diff;
    in_grads->at(4) = results->at(2);  // beta_diff
    if (is_training_) {
      in_grads->at(0) = results->at(0);
      return Maybe<void>::Ok();
    }

    DimVector dim_vec;
    for (int i = 0; i < x->shape()->NumAxes(); ++i) {
      if (i != axis_) {
        dim_vec.push_back(1);
      } else {
        dim_vec.push_back(x->shape()->At(axis_));
      }
    }
    const auto& reshape_op = JUST(op_expr_helper::ReshapeOp(Shape(dim_vec)));
    const auto& gamma_1 = JUST(Dispatch<Tensor>(*reshape_op, {gamma}));
    const auto& inv_variance_1 = JUST(Dispatch<Tensor>(*reshape_op, {inv_variance}));

    std::shared_ptr<Tensor> y_grad_fp32 = y_grad;
    bool is_fp16 = y_grad->dtype()->data_type() == DataType::kFloat16;
    if (is_fp16) { y_grad_fp32 = JUST(Dispatch<Tensor>(*h2f_cast_op_, {y_grad})); }
    const auto& dy_mul_gamma = JUST(Dispatch<Tensor>(*broadcast_mul_op_, {gamma_1, y_grad_fp32}));
    const auto& dy_mul_inv_var =
        JUST(Dispatch<Tensor>(*broadcast_mul_op_, {dy_mul_gamma, inv_variance_1}));
    if (is_fp16) {
      in_grads->at(0) = JUST(Dispatch<Tensor>(*f2h_cast_op_, {dy_mul_inv_var}));
    } else {
      in_grads->at(0) = dy_mul_inv_var;
    }
    return Maybe<void>::Ok();
  }

 private:
  std::string op_name_;
  int32_t axis_;
  bool is_training_;
  std::shared_ptr<OpExpr> add_eps_op_;
  std::shared_ptr<OpExpr> rsqrt_op_;
  std::shared_ptr<OpExpr> normalization_grad_op_;
  std::shared_ptr<OpExpr> broadcast_mul_op_;
  std::shared_ptr<OpExpr> h2f_cast_op_;
  std::shared_ptr<OpExpr> f2h_cast_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("normalization", NormalizationGrad);

}  // namespace one
}  // namespace oneflow
