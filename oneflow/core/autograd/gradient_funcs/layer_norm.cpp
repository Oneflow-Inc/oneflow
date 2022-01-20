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

struct LayerNormCaptureState : public AutoGradCaptureState {
  bool center = true;
  bool scale = true;

  int64_t begin_norm_axis = 1;
  int64_t begin_params_axis = 1;

  double epsilon = 1e-5;

  bool x_requires_grad = true;
  bool has_affine = true;

  size_t gamma_index = 0;
  size_t normalized_index = 1;
  size_t x_index = 2;
  size_t mean_index = 3;
  size_t inv_variance_index = 4;
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
};

Maybe<void> LayerNorm::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  op_name_ = fw_op_expr->op_name();
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

  bool has_normalized_diff = ctx->scale && inputs.at(0)->requires_grad();
  bool has_gamma_diff = ctx->scale && inputs.at(1)->requires_grad();
  bool has_beta_diff = ctx->center && inputs.at(2)->requires_grad();

  ctx->has_affine = has_normalized_diff && has_gamma_diff && has_beta_diff;

  if (ctx->has_affine) {
    ctx->gamma_index = ctx->SaveTensorForBackward(inputs.at(1));  // save gamma.
    ctx->normalized_index = ctx->SaveTensorForBackward(outputs.at(3));
  }

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

  if (!ctx->has_affine) {
    // Use LayerNormParamGrad(Tensor dy, Tensor gamma, Int64 begin_params_axis, Double epsilon).
    dy = JUST(functional::LayerNormParamGrad(dy, begin_params_axis, ctx->epsilon));
  } else {
    // Use LayerNormAffineParamGrad(Tensor dy, Tensor gamma, Tensor normalized, Int64
    // begin_params_axis, Double epsilon).
    std::shared_ptr<Tensor> gamma = saved_tensors.at(ctx->gamma_index);
    std::shared_ptr<Tensor> normalized = saved_tensors.at(ctx->normalized_index);
    const auto& results = JUST(functional::LayerNormAffineParamGrad(
        dy, gamma, normalized, begin_params_axis, ctx->epsilon));
    in_grads->at(1) = results->at(0);  // For gamma.
    in_grads->at(2) = results->at(1);  // For beta.
    dy = results->at(2);
  }

  if (ctx->x_requires_grad) {
    const auto& x = saved_tensors.at(ctx->x_index);
    const auto& mean = saved_tensors.at(ctx->mean_index);
    const auto& inv_variance = saved_tensors.at(ctx->inv_variance_index);
    in_grads->at(0) =
        JUST(functional::LayerNormGrad(x, mean, inv_variance, dy, begin_norm_axis, ctx->epsilon));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("layer_norm", LayerNorm);

}  // namespace one
}  // namespace oneflow
