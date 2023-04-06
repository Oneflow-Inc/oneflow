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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct GroupNormCaptureState : public AutoGradCaptureState {
  double epsilon = 1e-5;
  bool x_requires_grad = true;
  bool gamma_requires_grad = true;
  bool beta_requires_grad = true;
  bool affine = true;
  int32_t num_groups = 1;
  size_t x_index = 0;
  size_t mean_index = 1;
  size_t inv_variance_index = 2;
  size_t gamma_index = 3;
  std::string data_format;
  std::string activation;
};

class GroupNorm : public OpExprGradFunction<GroupNormCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(GroupNormCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;

  Maybe<void> Apply(const GroupNormCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::string op_name_;
};

Maybe<void> GroupNorm::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  op_name_ = fw_op_expr->op_name();
  return Maybe<void>::Ok();
}

Maybe<void> GroupNorm::Capture(GroupNormCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->affine = JUST(composed_attrs.GetAttr<bool>("affine"));
  ctx->epsilon = JUST(composed_attrs.GetAttr<double>("epsilon"));
  ctx->num_groups = JUST(composed_attrs.GetAttr<int32_t>("num_groups"));
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->activation = JUST(composed_attrs.GetAttr<std::string>("activation"));
  if (ctx->affine) {
    CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)
    ctx->gamma_requires_grad = inputs.at(1)->requires_grad();
    ctx->beta_requires_grad = inputs.at(2)->requires_grad();
  } else {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  }
  CHECK_EQ_OR_RETURN(outputs.size(), 3);  // NOLINT(maybe-need-error-msg)

  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  if (ctx->x_requires_grad || ctx->affine) {
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->mean_index = ctx->SaveTensorForBackward(outputs.at(1));
    ctx->inv_variance_index = ctx->SaveTensorForBackward(outputs.at(2));
    if (ctx->affine) {
      ctx->gamma_index = ctx->SaveTensorForBackward(inputs.at(1));  // save gamma.
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GroupNorm::Apply(const GroupNormCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(ctx->data_format, "channels_first");
  CHECK_EQ_OR_RETURN(ctx->activation, "none");
  const auto& saved_tensors = ctx->SavedTensors();
  if (ctx->affine) {
    in_grads->resize(3);
  } else {
    in_grads->resize(1);
  }
  const auto& dy = out_grads.at(0);
  const auto& x = saved_tensors.at(ctx->x_index);
  const auto& mean = saved_tensors.at(ctx->mean_index);
  const auto& inv_variance = saved_tensors.at(ctx->inv_variance_index);

  if (ctx->affine && (ctx->gamma_requires_grad || ctx->beta_requires_grad)) {
    const auto& results = JUST(functional::GroupNormParamGrad(dy, x, mean, inv_variance));
    if (ctx->gamma_requires_grad) { in_grads->at(1) = results->at(0); }  // For gamma.
    if (ctx->beta_requires_grad) { in_grads->at(2) = results->at(1); }   // For beta.
  }
  if (ctx->x_requires_grad) {
    if (ctx->affine) {
      std::shared_ptr<Tensor> gamma = saved_tensors.at(ctx->gamma_index);
      in_grads->at(0) = JUST(functional::GroupNormGrad(dy, x, mean, inv_variance, gamma,
                                                       ctx->num_groups, ctx->epsilon));
    } else {
      in_grads->at(0) = JUST(functional::GroupNormGrad(dy, x, mean, inv_variance, NullOpt,
                                                       ctx->num_groups, ctx->epsilon));
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("group_norm", GroupNorm);

}  // namespace one
}  // namespace oneflow
