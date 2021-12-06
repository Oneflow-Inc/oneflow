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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct NormalizationGradCaptureState : public AutoGradCaptureState {
  int32_t axis;
  float epsilon;
  bool track_running_stats;
  bool is_training;
  bool x_requires_grad;
  bool gamma_requires_grad;
  bool beta_requires_grad;
};

// training:
// y, mean, inv_variance = normalization(x, moving_mean, moving_variance, gamma, beta,
// axis=1, epsilon=0.01, momentum=0.9)
// y, mean, inv_variance = normalization(x, gamma, beta, axis=1, epsilon=0.01, momentum=0.9)

// inference:
// y = normalization(x, moving_mean, moving_variance, gamma, beta, axis=1, epsilon=0.01,
// momentum=0.9)
class NormalizationGrad : public OpExprGradFunction<NormalizationGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(NormalizationGradCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    // input_size may be 3 or 5, as inputs may be
    // (x, gamma, beta) or (x, moving_mean, moving_variance, gamma, beta)
    // ref to track_running_stats false/true
    // output_size may be 1 or 3, as outputs may be
    // (x, ) or (x, mean, inv_variance)
    // ref to is_training false/true
    state->x_requires_grad = inputs.at(0)->requires_grad();
    std::shared_ptr<Tensor> gamma, beta;
    if (inputs.size() == 3) {
      gamma = inputs.at(1);
      beta = inputs.at(2);
      state->track_running_stats = false;
    } else {
      CHECK_EQ_OR_RETURN(inputs.size(), 5);
      gamma = inputs.at(3);
      beta = inputs.at(4);
      state->track_running_stats = true;
    }
    state->gamma_requires_grad = gamma->requires_grad();
    state->beta_requires_grad = beta->requires_grad();
    auto* interp_ctx = dynamic_cast<const NormalizationGradOpInterpCtx*>(ctx);

    state->axis = interp_ctx->axis;
    state->epsilon = interp_ctx->epsilon;
    // state->is_training = interp_ctx->training;
    state->SaveTensorForBackward(inputs.at(0));  // x
    state->SaveTensorForBackward(gamma);         // gamma
    if (state->is_training || !state->track_running_stats) {
      state->SaveTensorForBackward(outputs.at(1));  // mean
      state->SaveTensorForBackward(outputs.at(2));  // inv_variance
    } else {
      state->SaveTensorForBackward(inputs.at(1));  // moving_mean
      state->SaveTensorForBackward(inputs.at(2));  // moving_variance
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NormalizationGradCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = state->SavedTensors().at(0);      // x
    const auto& gamma = state->SavedTensors().at(1);  // gamma
    const auto& y_grad = out_grads.at(0);

    std::shared_ptr<Tensor> mean, inv_variance;
    if (state->is_training || !state->track_running_stats) {
      mean = state->SavedTensors().at(2);          // mean
      inv_variance = state->SavedTensors().at(3);  // inv_variance
    } else {
      const auto& moving_mean = state->SavedTensors().at(2);      // moving_mean
      const auto& moving_variance = state->SavedTensors().at(3);  // moving_variance
      const auto& add_eps =
          JUST(functional::ScalarAdd(moving_variance, state->epsilon, /*inplace=*/false));
      mean = moving_mean;
      inv_variance = JUST(functional::Rsqrt(add_eps));
    }
    const auto& results = JUST(functional::NormalizationGrad(y_grad, x, mean, inv_variance, gamma,
                                                             state->epsilon, state->axis));
    CHECK_EQ_OR_RETURN(results->size(), 3);

    if (state->track_running_stats) {
      // The normalization op has 5 inputs which are x, moving_mean, moving_variance, gamma and
      // beta.
      in_grads->resize(5);
      if (state->gamma_requires_grad) {
        in_grads->at(3) = results->at(1);  // gamma_diff;
      }
      if (state->beta_requires_grad) {
        in_grads->at(4) = results->at(2);  // beta_diff
      }
    } else {
      // The normalization op has 3 inputs which are x, gamma and beta.
      in_grads->resize(3);
      if (state->gamma_requires_grad) {
        in_grads->at(1) = results->at(1);  // gamma_diff;
      }
      if (state->beta_requires_grad) {
        in_grads->at(2) = results->at(2);  // beta_diff
      }
    }

    if (!state->x_requires_grad) { return Maybe<void>::Ok(); }
    if (state->is_training) {
      in_grads->at(0) = results->at(0);
      return Maybe<void>::Ok();
    }

    DimVector dim_vec;
    for (int i = 0; i < x->shape()->NumAxes(); ++i) {
      if (i != state->axis) {
        dim_vec.push_back(1);
      } else {
        dim_vec.push_back(x->shape()->At(state->axis));
      }
    }
    Shape shape(dim_vec);
    const auto& reshaped_gamma = JUST(functional::Reshape(gamma, shape));
    const auto& reshaped_inv_variance = JUST(functional::Reshape(inv_variance, shape));

    std::shared_ptr<Tensor> y_grad_fp32 = y_grad;
    bool is_fp16 = y_grad->dtype()->data_type() == DataType::kFloat16;
    if (is_fp16) { y_grad_fp32 = JUST(functional::Cast(y_grad, DType::Float())); }
    const auto& dy_mul_gamma = JUST(functional::Mul(reshaped_gamma, y_grad_fp32));
    const auto& dy_mul_inv_var = JUST(functional::Mul(dy_mul_gamma, reshaped_inv_variance));
    if (is_fp16) {
      in_grads->at(0) = JUST(functional::Cast(dy_mul_inv_var, DType::Float16()));
    } else {
      in_grads->at(0) = dy_mul_inv_var;
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("normalization", NormalizationGrad);

}  // namespace one
}  // namespace oneflow
