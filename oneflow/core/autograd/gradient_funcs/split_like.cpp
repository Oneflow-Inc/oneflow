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

struct SplitLikeCaptureState : public AutoGradCaptureState {
  int64_t max_dim_size;
  bool requires_grad;
};

class SplitLike : public OpExprGradFunction<SplitLikeCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SplitLikeCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SplitLikeCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  int64_t axis_;
  std::vector<std::shared_ptr<OpExpr>> zero_like_ops_;
  std::shared_ptr<OpExpr> concat_op_;
};

Maybe<void> SplitLike::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());
  axis_ = JUST(op_trait_->GetAttr<int64_t>("axis"));
  int32_t output_num = JUST(op_trait_->output_size("out"));
  concat_op_ = JUST(
      op_expr_helper::ConcatOp(output_num, axis_, /*max_dim_size=*/-1, GradientOpName(op_name)));
  zero_like_ops_.resize(output_num);
  for (int i = 0; i < output_num; ++i) {
    zero_like_ops_[i] = JUST(
        op_expr_helper::ZeroLikeOp(GradientOpName(op_name + "_zero_like" + std::to_string(i))));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Capture(SplitLikeCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), outputs.size() + 1);
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->max_dim_size = 0;
  for (int i = 0; i < outputs.size(); ++i) {
    ctx->max_dim_size += inputs.at(i + 1)->shape()->At(axis_);
    ctx->SaveTensorForBackward(outputs.at(i));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLike::Apply(const SplitLikeCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  in_grads->resize(1);
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  CHECK_EQ_OR_RETURN(out_grads.size(), zero_like_ops_.size());
  const auto& saved_tensors = ctx->SavedTensors();
  TensorTuple inputs;
  inputs.reserve(out_grads.size());
  for (int i = 0; i < out_grads.size(); ++i) {
    const auto& out_grad_i = out_grads.at(i);
    if (out_grad_i.get()) {
      inputs.push_back(out_grad_i);
    } else {
      const auto& zero_grad =
          JUST(OpInterpUtil::Dispatch<Tensor>(*zero_like_ops_.at(i), {saved_tensors.at(i)}));
      inputs.push_back(zero_grad);
    }
  }
  MutableAttrMap concat_attrs;
  JUST(concat_attrs.SetAttr<int>("axis", axis_));
  JUST(concat_attrs.SetAttr<int>("max_dim_size", ctx->max_dim_size));
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*concat_op_, inputs, concat_attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("split_like", SplitLike);

}  // namespace one
}  // namespace oneflow
