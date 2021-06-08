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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/user_op_conf_trait.h"

namespace oneflow {
namespace one {

namespace {

struct ConvInterpState : public OpExprInterpState {
  bool weight_requires_grad = true;
  bool input_requires_grad = true;
};

class ConvNdGrad : public OpExprGradFunction<ConvInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ConvInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const ConvInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  std::shared_ptr<std::string> data_format_;
  std::shared_ptr<std::vector<int32_t>> padding_before_;
  std::shared_ptr<std::vector<int32_t>> kernel_size_;
  std::shared_ptr<std::vector<int32_t>> strides_;
  std::shared_ptr<std::vector<int32_t>> dilation_rate_;
  int32_t groups_;

  std::shared_ptr<OpExpr> data_grad_op_;
  std::shared_ptr<OpExpr> weight_grad_op_;
};

Maybe<void> ConvNdGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());

  data_format_ = JUST(op_trait_->GetAttr<std::string>("data_format"));
  padding_before_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("padding_before"));
  kernel_size_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("kernel_size"));
  strides_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("strides"));
  dilation_rate_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("dilation_rate"));
  groups_ = JUST(op_trait_->GetAttr<int32_t>("groups"));
  int32_t ndims = kernel_size_->size();
  CHECK_EQ_OR_RETURN(ndims, strides_->size());
  CHECK_EQ_OR_RETURN(ndims, dilation_rate_->size());
  data_grad_op_ = JUST(op_expr_helper::ConvNdDataGradOp(*kernel_size_, *strides_, *padding_before_,
                                                        *dilation_rate_, groups_, *data_format_));

  weight_grad_op_ = JUST(op_expr_helper::ConvNdFilterGradOp(
      *kernel_size_, *strides_, *padding_before_, *dilation_rate_, groups_, *data_format_));

  return Maybe<void>::Ok();
}

Maybe<void> ConvNdGrad::Capture(ConvInterpState* ctx, const TensorTuple& inputs,
                                const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  ctx->weight_requires_grad = inputs.at(1)->requires_grad();
  ctx->SaveTensorForBackward(inputs.at(0));  // x
  if (ctx->input_requires_grad) {
    ctx->SaveTensorForBackward(inputs.at(1));  // weight
  }
  return Maybe<void>::Ok();
}

Maybe<void> ConvNdGrad::Apply(const ConvInterpState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);

  in_grads->resize(2);
  if (ctx->input_requires_grad) {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& weight = ctx->SavedTensors().at(1);
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*data_grad_op_, {dy, weight, x}, /*attrs=*/{}));
  }
  if (ctx->weight_requires_grad) {
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->at(1) = JUST(OpInterpUtil::Dispatch<Tensor>(*weight_grad_op_, {dy, x}, /*attrs=*/{}));
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_EXPR_GRAD_FUNCTION("conv1d", ConvNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv2d", ConvNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv3d", ConvNdGrad);

}  // namespace one
}  // namespace oneflow
