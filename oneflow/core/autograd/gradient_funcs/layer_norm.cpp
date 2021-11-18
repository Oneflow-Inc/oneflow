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
  int64_t begin_norm_axis = 1;

  bool x_requires_grad = true;
  bool gamma_beta_requires_grad = true;

  size_t gamma_index = 0;
  size_t x_index = 1;
  size_t mean_index = 2;
  size_t inv_variance_index = 3;
};

// y, mean, inv_variance, [normalized] =
//   layer_norm(x, [beta], [gamma], begin_norm_axis=1, epsilon=1e-5)
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
  ctx->begin_norm_axis = JUST(composed_attrs.GetAttr<int64_t>("begin_norm_axis"));
  bool has_affine = false;
  if (inputs.size() == 3) {
    has_affine = true;
  } else {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
  }
  CHECK_EQ_OR_RETURN(outputs.size(), 3);

  bool has_gamma_diff = has_affine && inputs.at(1)->requires_grad();
  bool has_beta_diff = has_affine && inputs.at(2)->requires_grad();
  CHECK_EQ_OR_RETURN(has_gamma_diff, has_beta_diff);

  ctx->gamma_beta_requires_grad = has_gamma_diff && has_gamma_diff;
  ctx->x_requires_grad = inputs.at(0)->requires_grad();

  if (ctx->x_requires_grad || ctx->gamma_beta_requires_grad) {
    if (has_affine) { ctx->gamma_index = ctx->SaveTensorForBackward(inputs.at(1)); }
    ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    ctx->mean_index = ctx->SaveTensorForBackward(outputs.at(1));
    ctx->inv_variance_index = ctx->SaveTensorForBackward(outputs.at(2));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Apply(const LayerNormCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& saved_tensors = ctx->SavedTensors();
  int in_grad_size = 0;
  if (ctx->x_requires_grad) { in_grad_size += 1; }
  if (ctx->gamma_beta_requires_grad) { in_grad_size += 2; }
  in_grads->resize(in_grad_size);
  std::shared_ptr<Tensor> dy = out_grads.at(0);
  int64_t begin_norm_axis = ctx->begin_norm_axis;
  if (begin_norm_axis < 0) { begin_norm_axis += dy->shape()->NumAxes(); }

  const auto& x = saved_tensors.at(ctx->x_index);
  const auto& mean = saved_tensors.at(ctx->mean_index);
  const auto& inv_variance = saved_tensors.at(ctx->inv_variance_index);
  if (ctx->x_requires_grad && ctx->gamma_beta_requires_grad) {
    std::shared_ptr<Tensor> gamma = saved_tensors.at(ctx->gamma_index);
    const auto& results =
        JUST(functional::LayerNormAffineGrad(dy, x, mean, inv_variance, gamma, begin_norm_axis));
    in_grads->at(0) = results->at(0);  // For dx.
    in_grads->at(1) = results->at(1);  // For gamma.
    in_grads->at(2) = results->at(2);  // For gamma.
  } else if (ctx->x_requires_grad) {
    in_grads->at(0) = JUST(functional::LayerNormGrad(dy, x, mean, inv_variance, begin_norm_axis));
  } else {
    UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("layer_norm", LayerNorm);

}  // namespace one
}  // namespace oneflow
