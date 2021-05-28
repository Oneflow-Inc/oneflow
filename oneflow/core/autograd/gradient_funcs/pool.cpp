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

class MaxPoolNd : public OpExprGradFunction<OpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  std::shared_ptr<std::string> data_format_;
  std::shared_ptr<std::string> padding_;
  std::shared_ptr<std::vector<int32_t>> padding_before_;
  std::shared_ptr<std::vector<int32_t>> padding_after_;
  std::shared_ptr<std::vector<int32_t>> pool_size_;
  std::shared_ptr<std::vector<int32_t>> strides_;
  bool ceil_mode_;

  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> MaxPoolNd::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());

  data_format_ = JUST(op_trait_->GetAttr<std::string>("data_format"));
  padding_ = JUST(op_trait_->GetAttr<std::string>("padding"));
  padding_before_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("padding_before"));
  padding_after_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("padding_after"));
  pool_size_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("pool_size"));
  strides_ = JUST(op_trait_->GetAttr<std::vector<int32_t>>("strides"));
  ceil_mode_ = JUST(op_trait_->GetAttr<bool>("ceil_mode"));
  const std::string mode = "max";
  int32_t ndims = pool_size_->size();
  CHECK_EQ_OR_RETURN(ndims, strides_->size());
  CHECK_EQ_OR_RETURN(ndims, padding_before_->size());
  CHECK_EQ_OR_RETURN(ndims, padding_after_->size());
  grad_op_ = JUST(op_expr_helper::PoolNdOp(mode, *data_format_, *padding_, *padding_before_,
                                           *padding_after_, *pool_size_, *strides_, ceil_mode_));
  return Maybe<void>::Ok();
}

Maybe<void> MaxPoolNd::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> MaxPoolNd::Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& saved_tensors = ctx->SavedTensors();
  in_grads->resize(1);
  in_grads->at(0) =
      JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0), saved_tensors.at(0)}, /*attrs=*/{}));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_1d", MaxPoolNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_2d", MaxPoolNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_3d", MaxPoolNd);

}  // namespace one
}  // namespace oneflow
