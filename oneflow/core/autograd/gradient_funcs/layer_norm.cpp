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
#include "oneflow/core/framework/attr_map.h"

namespace oneflow {
namespace one {

struct LayerNormCaptureState : public AutoGradCaptureState {
  bool center;
  bool scale;

  int64_t begin_norm_axis;
  int64_t begin_params_axis;

  double epsilon;

  bool x_requires_grad;
  bool has_beta_diff;
  bool has_gamma_diff;
  bool has_normalized_diff;

  size_t gamma_index;
  size_t normalized_index;
  size_t x_index;
  size_t mean_index;
  size_t inv_variance_index;
};

// y, mean, inv_variance, [normalized] =
//   layer_norm(x, [beta], [gamma], center=False, scale=False, begin_norm_axis=1,
//              begin_params_axis=-1, epsilon=1e-5)
class LayerNorm : public OpExprGradFunction<LayerNormCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(LayerNormCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;

  Maybe<void> Apply(const LayerNormCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::string op_name_;
  std::shared_ptr<OpExpr> x_grad_op_;
};

Maybe<void> LayerNorm::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  op_name_ = fw_op_expr->op_name();
  x_grad_op_ = JUST(op_expr_helper::LayerNormGradOp(/*begin_norm_axis=*/1, /*epsilon=*/1e-5,
                                                    GradientOpName(op_name_)));
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Capture(LayerNormCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->center = JUST(composed_attrs.GetAttr<bool>("center"));
  ctx->scale = JUST(composed_attrs.GetAttr<bool>("scale"));
  ctx->begin_norm_axis = JUST(composed_attrs.GetAttr<int64_t>("begin_norm_axis"));
  ctx->begin_params_axis = JUST(composed_attrs.GetAttr<int64_t>("begin_params_axis"));
  ctx->epsilon = JUST(composed_attrs.GetAttr<double>("epsilon"));

  CHECK_EQ_OR_RETURN(inputs.size(), ctx->center + ctx->scale + 1);
  CHECK_EQ_OR_RETURN(outputs.size(), ctx->scale + 3);
  ctx->has_beta_diff = ctx->center && inputs.at(1)->requires_grad();
  const int gamma_index = ctx->center + 1;
  ctx->has_gamma_diff = ctx->scale && inputs.at(gamma_index)->requires_grad();
  ctx->has_normalized_diff = ctx->scale && inputs.at(0)->requires_grad();
  if (ctx->has_gamma_diff || ctx->has_normalized_diff) {
    ctx->gamma_index = ctx->SaveTensorForBackward(inputs.at(gamma_index));
  }
  if (ctx->has_gamma_diff) { ctx->normalized_index = ctx->SaveTensorForBackward(outputs.at(3)); }
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  if (ctx->x_requires_grad) {
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->mean_index = ctx->SaveTensorForBackward(outputs.at(1));
    ctx->inv_variance_index = ctx->SaveTensorForBackward(outputs.at(2));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Apply(const LayerNormCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& saved_tensors = ctx->SavedTensors();
  in_grads->resize(ctx->center + ctx->scale + 1);
  std::shared_ptr<Tensor> dy = out_grads.at(0);
  int64_t begin_params_axis = ctx->begin_params_axis;
  if (begin_params_axis < 0) { begin_params_axis += dy->shape()->NumAxes(); }
  int64_t begin_norm_axis = ctx->begin_norm_axis;
  if (begin_norm_axis < 0) { begin_norm_axis += dy->shape()->NumAxes(); }
  if (ctx->has_beta_diff || ctx->has_gamma_diff || ctx->has_normalized_diff) {
    const auto& param_grad_op = JUST(op_expr_helper::LayerNormParamGradOp(
        begin_params_axis, ctx->has_beta_diff, ctx->has_gamma_diff, ctx->has_normalized_diff,
        GradientOpName(op_name_ + "_param")));
    TensorTuple inputs{dy};
    if (ctx->has_gamma_diff || ctx->has_normalized_diff) {
      inputs.push_back(saved_tensors.at(ctx->gamma_index));  // gamma
    }
    if (ctx->has_gamma_diff) {
      inputs.push_back(saved_tensors.at(ctx->normalized_index));  // normalized
    }
    const auto& results = JUST(OpInterpUtil::Dispatch<TensorTuple>(*param_grad_op, inputs));
    if (ctx->has_beta_diff) { in_grads->at(1) = results->at(0); }
    if (ctx->has_gamma_diff) {
      in_grads->at(ctx->has_beta_diff + 1) = results->at(ctx->has_beta_diff);
    }
    if (ctx->has_normalized_diff) { dy = results->at(ctx->has_beta_diff + ctx->has_gamma_diff); }
  }
  if (ctx->x_requires_grad) {
    const auto& x = saved_tensors.at(ctx->x_index);
    const auto& mean = saved_tensors.at(ctx->mean_index);
    const auto& inv_variance = saved_tensors.at(ctx->inv_variance_index);
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<double>("epsilon", ctx->epsilon));
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*x_grad_op_, {x, mean, inv_variance, dy}, attrs));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("layer_norm", LayerNorm);

}  // namespace one
}  // namespace oneflow
