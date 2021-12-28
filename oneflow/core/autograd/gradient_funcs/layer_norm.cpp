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
#include "oneflow/core/framework/op_generated.h"
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
  Maybe<void> Capture(LayerNormCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;

  Maybe<void> Apply(const LayerNormCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> LayerNorm::Capture(LayerNormCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpBase* ctx) const {
  auto* op_ctx = dynamic_cast<const LayerNormOp*>(ctx);
  state->center = op_ctx->center();
  state->scale = op_ctx->scale();
  state->begin_norm_axis = op_ctx->begin_norm_axis();
  state->begin_params_axis = op_ctx->begin_params_axis();
  state->epsilon = op_ctx->epsilon();

  CHECK_EQ_OR_RETURN(inputs.size(), state->center + state->scale + 1);
  CHECK_EQ_OR_RETURN(outputs.size(), state->scale + 3);

  bool has_normalized_diff = state->scale && inputs.at(0)->requires_grad();
  bool has_gamma_diff = state->scale && inputs.at(1)->requires_grad();
  bool has_beta_diff = state->center && inputs.at(2)->requires_grad();

  state->has_affine = has_normalized_diff && has_gamma_diff && has_beta_diff;

  if (state->has_affine) {
    state->gamma_index = state->SaveTensorForBackward(inputs.at(1));  // save gamma.
    state->normalized_index = state->SaveTensorForBackward(outputs.at(3));
  }

  state->x_requires_grad = inputs.at(0)->requires_grad();
  if (state->x_requires_grad) {
    state->x_index = state->SaveTensorForBackward(inputs.at(0));
    state->mean_index = state->SaveTensorForBackward(outputs.at(1));
    state->inv_variance_index = state->SaveTensorForBackward(outputs.at(2));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNorm::Apply(const LayerNormCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& saved_tensors = state->SavedTensors();
  in_grads->resize(state->center + state->scale + 1);
  std::shared_ptr<Tensor> dy = out_grads.at(0);
  int64_t begin_params_axis = state->begin_params_axis;
  if (begin_params_axis < 0) { begin_params_axis += dy->shape()->NumAxes(); }
  int64_t begin_norm_axis = state->begin_norm_axis;
  if (begin_norm_axis < 0) { begin_norm_axis += dy->shape()->NumAxes(); }

  std::shared_ptr<Tensor> gamma = saved_tensors.at(state->gamma_index);
  if (!state->has_affine) {
    // Use LayerNormParamGrad(Tensor dy, Tensor gamma, Int64 begin_params_axis, Double epsilon).
    dy = JUST(functional::LayerNormParamGrad(dy, begin_params_axis, state->epsilon));
  } else {
    // Use LayerNormAffineParamGrad(Tensor dy, Tensor gamma, Tensor normalized, Int64
    // begin_params_axis, Double epsilon).
    std::shared_ptr<Tensor> normalized = saved_tensors.at(state->normalized_index);
    const auto& results = JUST(functional::LayerNormAffineParamGrad(
        dy, gamma, normalized, begin_params_axis, state->epsilon));
    in_grads->at(1) = results->at(0);  // For gamma.
    in_grads->at(2) = results->at(1);  // For beta.
    dy = results->at(2);
  }

  if (state->x_requires_grad) {
    const auto& x = saved_tensors.at(state->x_index);
    const auto& mean = saved_tensors.at(state->mean_index);
    const auto& inv_variance = saved_tensors.at(state->inv_variance_index);
    in_grads->at(0) =
        JUST(functional::LayerNormGrad(x, mean, inv_variance, dy, begin_norm_axis, state->epsilon));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("layer_norm", LayerNorm);

}  // namespace one
}  // namespace oneflow
