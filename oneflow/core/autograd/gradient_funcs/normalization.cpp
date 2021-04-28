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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/user_op_conf_trait.h"

namespace oneflow {
namespace one {

struct NormalizationGradInterpState : public OpExprInterpState {
  bool is_training;
};

// training:
// y, mean, inv_variance = normalization(x, moving_mean, moving_variance, gamma, beta,
// axis=1, epsilon=0.01, momentum=0.9)
// inference:
// y = normalization(x, moving_mean, moving_variance, gamma, beta, axis=1, epsilon=0.01,
// momentum=0.9)
class NormalizationGrad : public OpExprGradFunction<NormalizationGradInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    const std::string& op_name = fw_op_expr->op_name();
    op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());
    const float epsilon = JUST(op_trait_->GetAttr<float>("epsilon"));
    axis_ = JUST(op_trait_->GetAttr<int32_t>("axis"));
    // v1 = variance + eps
    add_eps_op_ = JUST(op_expr_helper::ScalarAddOp(epsilon, GradientOpName(op_name + "_add_eps")));
    // v2 = rsqrt(v1)
    rsqrt_op_ = JUST(op_expr_helper::RsqrtOp(GradientOpName(op_name + "_rsqrt")));

    // Normalization grad.
    normalization_grad_op_ = JUST(op_expr_helper::NormalizationGradOp(
        axis_, epsilon, GradientOpName(op_name + "_norm_grad")));

    reshape_gamma_op_ =
        JUST(op_expr_helper::ReshapeOp(Shape{-1}, GradientOpName(op_name + "_reshape_gamma")));
    reshape_variance_op_ =
        JUST(op_expr_helper::ReshapeOp(Shape{-1}, GradientOpName(op_name + "_reshape_variance")));

    broadcast_mul_op_ = JUST(op_expr_helper::BroadcastMulOp(GradientOpName(op_name + "_mul")));
    // fp16 -> fp32 and fp32 -> fp16.
    h2f_cast_op_ = JUST(op_expr_helper::CastOp(DataType::kFloat, GradientOpName(op_name + "_h2f")));
    f2h_cast_op_ =
        JUST(op_expr_helper::CastOp(DataType::kFloat16, GradientOpName(op_name + "_f2h")));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(NormalizationGradInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrValueMap& attrs) const override {
    ctx->is_training = JUST(op_trait_->GetAttr<bool>("training", attrs));
    ctx->SaveTensorForBackward(inputs.at(0));  // x
    ctx->SaveTensorForBackward(inputs.at(3));  // gamma
    if (ctx->is_training) {
      ctx->SaveTensorForBackward(outputs.at(1));  // mean
      ctx->SaveTensorForBackward(outputs.at(2));  // inv_variance
    } else {
      ctx->SaveTensorForBackward(inputs.at(1));  // moving_mean
      ctx->SaveTensorForBackward(inputs.at(2));  // moving_variance
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NormalizationGradInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);      // x
    const auto& gamma = ctx->SavedTensors().at(1);  // gamma
    const auto& y_grad = out_grads.at(0);

    std::shared_ptr<Tensor> mean, inv_variance;
    if (ctx->is_training) {
      mean = ctx->SavedTensors().at(2);          // mean
      inv_variance = ctx->SavedTensors().at(3);  // inv_variance
    } else {
      const auto& moving_mean = ctx->SavedTensors().at(2);      // moving_mean
      const auto& moving_variance = ctx->SavedTensors().at(3);  // moving_variance
      const auto& add_eps = JUST(OpInterpUtil::Dispatch<Tensor>(*add_eps_op_, {moving_variance}));
      mean = moving_mean;
      inv_variance = JUST(OpInterpUtil::Dispatch<Tensor>(*rsqrt_op_, {add_eps}));
    }
    const auto& results = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *normalization_grad_op_, {x, y_grad, gamma, mean, inv_variance}));
    CHECK_EQ_OR_RETURN(results->size(), 3);
    // The normalization op has 5 inputs which are x, moving_mean, moving_variance, gamma and beta.
    in_grads->resize(5);
    in_grads->at(3) = results->at(1);  // gamma_diff;
    in_grads->at(4) = results->at(2);  // beta_diff
    if (ctx->is_training) {
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
    MutableAttrValueMap shape_attr;
    shape_attr.SetAttr<Shape>("shape", Shape(dim_vec));
    const auto& reshaped_gamma =
        JUST(OpInterpUtil::Dispatch<Tensor>(*reshape_gamma_op_, {gamma}, shape_attr));
    const auto& reshaped_inv_variance =
        JUST(OpInterpUtil::Dispatch<Tensor>(*reshape_variance_op_, {inv_variance}, shape_attr));

    std::shared_ptr<Tensor> y_grad_fp32 = y_grad;
    bool is_fp16 = y_grad->dtype()->data_type() == DataType::kFloat16;
    if (is_fp16) { y_grad_fp32 = JUST(OpInterpUtil::Dispatch<Tensor>(*h2f_cast_op_, {y_grad})); }
    const auto& dy_mul_gamma =
        JUST(OpInterpUtil::Dispatch<Tensor>(*broadcast_mul_op_, {reshaped_gamma, y_grad_fp32}));
    const auto& dy_mul_inv_var = JUST(
        OpInterpUtil::Dispatch<Tensor>(*broadcast_mul_op_, {dy_mul_gamma, reshaped_inv_variance}));
    if (is_fp16) {
      in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*f2h_cast_op_, {dy_mul_inv_var}));
    } else {
      in_grads->at(0) = dy_mul_inv_var;
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  int32_t axis_;
  std::shared_ptr<OpExpr> add_eps_op_;
  std::shared_ptr<OpExpr> rsqrt_op_;
  std::shared_ptr<OpExpr> normalization_grad_op_;
  std::shared_ptr<OpExpr> reshape_gamma_op_;
  std::shared_ptr<OpExpr> reshape_variance_op_;
  std::shared_ptr<OpExpr> broadcast_mul_op_;
  std::shared_ptr<OpExpr> h2f_cast_op_;
  std::shared_ptr<OpExpr> f2h_cast_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("normalization", NormalizationGrad);

}  // namespace one
}  // namespace oneflow
