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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct NormalizationAddReluGradCaptureState : public AutoGradCaptureState {
  int32_t axis = 1;
  float epsilon = 1e-5;
  bool track_running_stats = true;
  bool is_training = true;
  bool has_addend = false;
  bool x_requires_grad = true;
  bool addend_requires_grad = true;
  bool gamma_requires_grad = true;
  bool beta_requires_grad = true;
};

// training:
// y, mean, inv_variance = normalization_add_relu(x, Optional(add_end), moving_mean,
// moving_variance, gamma, beta, axis=1, epsilon=0.01, momentum=0.9) y, mean, inv_variance =
// normalization_add_relu(x, Optional(add_end), gamma, beta, axis=1, epsilon=0.01, momentum=0.9)

// inference:
// y = normalization_add_relu(x, Optional(add_end), moving_mean, moving_variance, gamma, beta,
// axis=1, epsilon=0.01, momentum=0.9)

class NormalizationAddReluGrad : public OpExprGradFunction<NormalizationAddReluGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(NormalizationAddReluGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // input_size may be 3/4/5/6, as inputs may be
    // (x, gamma, beta) or (x, moving_mean, moving_variance, gamma, beta)
    // (x, addend, gamma, beta) or (x, addend, moving_mean, moving_variance, gamma, beta)

    // ref to track_running_stats false/true
    // output_size may be 2 or 4, as outputs may be
    // (x, reserve_space) or (x, reserve_space, mean, inv_variance)
    // ref to is_training false/true
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    std::shared_ptr<Tensor> add_end, gamma, beta;

    if (inputs.size() == 3 || inputs.size() == 5) {
      add_end = nullptr;
      if (inputs.size() == 3) {
        gamma = inputs.at(1);
        beta = inputs.at(2);
        ctx->track_running_stats = false;
      } else {
        gamma = inputs.at(3);
        beta = inputs.at(4);
        ctx->track_running_stats = true;
      }
      ctx->has_addend = false;
    } else if (inputs.size() == 4 || inputs.size() == 6) {
      add_end = inputs.at(1);
      if (inputs.size() == 4) {
        gamma = inputs.at(2);
        beta = inputs.at(3);
        ctx->track_running_stats = false;
      } else {
        gamma = inputs.at(4);
        beta = inputs.at(5);
        ctx->track_running_stats = true;
      }
      ctx->has_addend = true;
      ctx->addend_requires_grad = inputs.at(1)->requires_grad();
    }

    ctx->gamma_requires_grad = gamma->requires_grad();
    ctx->beta_requires_grad = beta->requires_grad();
    ComposedAttrMap composed_attrs(attrs, base_attrs_);

    ctx->axis = JUST(composed_attrs.GetAttr<int32_t>("axis"));
    ctx->epsilon = JUST(composed_attrs.GetAttr<float>("epsilon"));
    ctx->is_training = JUST(composed_attrs.GetAttr<bool>("training"));

    ctx->SaveTensorForBackward(inputs.at(0));  // x 0
    ctx->SaveTensorForBackward(gamma);         // gamma 1
    ctx->SaveTensorForBackward(beta);          // beta 2

    if (ctx->is_training || !ctx->track_running_stats) {
      ctx->SaveTensorForBackward(outputs.at(2));  // mean 3
      ctx->SaveTensorForBackward(outputs.at(3));  // inv_variance 4
    } else {
      if (inputs.size() == 5) {
        // without add_end
        ctx->SaveTensorForBackward(inputs.at(1));  // moving_mean 3
        ctx->SaveTensorForBackward(inputs.at(2));  // moving_variance 4
      } else {
        CHECK_EQ_OR_RETURN(inputs.size(), 6);  // NOLINT(maybe-need-error-msg)
        // with add_end
        ctx->SaveTensorForBackward(inputs.at(2));  // moving_mean 3
        ctx->SaveTensorForBackward(inputs.at(3));  // moving_variance 4
      }
    }
    ctx->SaveTensorForBackward(outputs.at(0));  // y 5
    ctx->SaveTensorForBackward(outputs.at(1));  // reserve space 6

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NormalizationAddReluGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);      // x
    const auto& gamma = ctx->SavedTensors().at(1);  // gamma
    const auto& beta = ctx->SavedTensors().at(2);   // beta
    const auto& y_grad = out_grads.at(0);

    std::shared_ptr<Tensor> mean, inv_variance;
    if (ctx->is_training || !ctx->track_running_stats) {
      mean = ctx->SavedTensors().at(3);          // mean
      inv_variance = ctx->SavedTensors().at(4);  // inv_variance
    } else {
      const auto& moving_mean = ctx->SavedTensors().at(3);      // moving_mean
      const auto& moving_variance = ctx->SavedTensors().at(4);  // moving_variance
      const auto& add_eps = JUST(
          functional::ScalarAdd(moving_variance, ctx->epsilon, /*alpha=*/1, /*inplace=*/false));
      mean = moving_mean;
      inv_variance = JUST(functional::Rsqrt(add_eps));
    }
    const auto& y = ctx->SavedTensors().at(5);
    const auto& reserve_space = ctx->SavedTensors().at(6);

    const auto& results = JUST(functional::NormalizationAddReluGrad(
        x, y_grad, mean, inv_variance, gamma, beta, reserve_space, y, ctx->axis, ctx->epsilon,
        ctx->has_addend));
    CHECK_EQ_OR_RETURN(results->size(), (ctx->has_addend ? 4 : 3))
        << Error::RuntimeError() << "The number of results is expected to be "
        << (ctx->has_addend ? 4 : 3) << ", but got "
        << results->size();  // here output includes "gamma_diff" "beta_diff" "dx" "addend_diff"

    if (ctx->track_running_stats) {
      // The normalization op has 5 inputs which are x, moving_mean, moving_variance, gamma and
      // beta. or 6 inputs: x, add_end, moving_mean, moving_variance, gamma and beta.
      if (ctx->has_addend) {
        in_grads->resize(6);
        if (ctx->gamma_requires_grad) {
          in_grads->at(4) = results->at(1);  // gamma_diff;
        }
        if (ctx->beta_requires_grad) {
          in_grads->at(5) = results->at(2);  // beta_diff
        }
        if (ctx->addend_requires_grad) {
          in_grads->at(1) = results->at(3);  // add_end_diff
        }
      } else {
        in_grads->resize(5);
        if (ctx->gamma_requires_grad) {
          in_grads->at(3) = results->at(1);  // gamma_diff;
        }
        if (ctx->beta_requires_grad) {
          in_grads->at(4) = results->at(2);  // beta_diff
        }
      }

    } else {
      // The normalization op has 3 inputs which are x, addend, gamma and beta.
      // or has 4 inputs which are x, addend, gamma and beta.
      if (ctx->has_addend) {
        in_grads->resize(4);
        if (ctx->addend_requires_grad) {
          in_grads->at(1) = results->at(3);  // addend_diff
        }
        if (ctx->gamma_requires_grad) {
          in_grads->at(2) = results->at(1);  // gamma_diff;
        }
        if (ctx->beta_requires_grad) {
          in_grads->at(3) = results->at(2);  // beta_diff
        }
      } else {
        in_grads->resize(3);
        if (ctx->gamma_requires_grad) {
          in_grads->at(1) = results->at(1);  // gamma_diff;
        }
        if (ctx->beta_requires_grad) {
          in_grads->at(2) = results->at(2);  // beta_diff
        }
      }
    }

    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    if (ctx->is_training) {
      in_grads->at(0) = results->at(0);
      return Maybe<void>::Ok();
    }

    // todo(zzk): add eval mode.
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("normalization_add_relu", NormalizationAddReluGrad);

}  // namespace one
}  // namespace oneflow
