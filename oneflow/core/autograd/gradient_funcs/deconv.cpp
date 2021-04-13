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

class DeConvolutionNd : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string op_name_;
  std::string data_format_;
  std::vector<int32_t> padding_before_;
  std::vector<int32_t> kernel_size_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> dilation_rate_;
  mutable bool activation_requires_grad_;
  mutable bool weight_requires_grad_;

  std::shared_ptr<OpExpr> activation_grad_op_;
  std::shared_ptr<OpExpr> weight_grad_op_;
};

Maybe<void> DeConvolutionNd::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  op_name_ = fw_op_expr->op_name();
  data_format_ = GetAttr<std::string>(fw_op_expr->proto(), "data_format");
  padding_before_ = GetAttr<std::vector<int32_t>>(fw_op_expr->proto(), "padding_before");
  kernel_size_ = GetAttr<std::vector<int32_t>>(fw_op_expr->proto(), "kernel_size");
  strides_ = GetAttr<std::vector<int32_t>>(fw_op_expr->proto(), "strides");
  dilation_rate_ = GetAttr<std::vector<int32_t>>(fw_op_expr->proto(), "dilation_rate");
  int32_t ndims = kernel_size_.size();
  CHECK_EQ_OR_RETURN(ndims, strides_.size());
  CHECK_EQ_OR_RETURN(ndims, dilation_rate_.size());
  int32_t filters = GetAttr<int32_t>(fw_op_expr->proto(), "filters");
  activation_grad_op_ =
      JUST(op_expr_helper::ConvNdOp(filters, kernel_size_, strides_, padding_before_,
                                    dilation_rate_, /*groups=*/1, data_format_));
  weight_grad_op_ = JUST(op_expr_helper::ConvNdFilterGradOp(
      kernel_size_, strides_, padding_before_, dilation_rate_, /*groups=*/1, data_format_));
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs) const {
  activation_requires_grad_ = inputs.at(0)->requires_grad();
  weight_requires_grad_ = inputs.at(1)->requires_grad();
  if (activation_requires_grad_) {
    ctx->SaveTensorForBackward(inputs.at(1));  // weight
  }
  if (weight_requires_grad_) {
    ctx->SaveTensorForBackward(inputs.at(0));  // x
  }
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (activation_requires_grad_) {
    const auto& weight = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(Dispatch<Tensor>(*activation_grad_op_, {out_grads.at(0), weight}));
  }
  if (weight_requires_grad_) {
    const auto& x = ctx->SavedTensors().at(activation_requires_grad_);
    in_grads->at(1) = JUST(Dispatch<Tensor>(*weight_grad_op_, {x, out_grads.at(0)}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("deconv1d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv2d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv3d", DeConvolutionNd);

}  // namespace one
}  // namespace oneflow
