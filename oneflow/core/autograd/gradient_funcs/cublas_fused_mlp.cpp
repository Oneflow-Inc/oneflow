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
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace one {

struct CublasFusedMLPCaptureState : public AutoGradCaptureState {
  int32_t weight_num = 0;
  bool skip_final_activation = false;
  bool x_requires_grad = false;
  std::vector<bool> weights_requires_grad;
  std::vector<bool> biases_requires_grad;
};

class CublasFusedMLP : public OpExprGradFunction<CublasFusedMLPCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(CublasFusedMLPCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const CublasFusedMLPCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> CublasFusedMLP::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> CublasFusedMLP::Capture(CublasFusedMLPCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() % 2 == 1)
      << Error::RuntimeError() << "Both weight and bias should be passed together";
  int32_t weight_num = (inputs.size() - 1) / 2;
  ctx->weight_num = weight_num;
  ctx->x_requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
  ctx->weights_requires_grad.resize(weight_num);
  ctx->biases_requires_grad.resize(weight_num);

  for (int32_t i = 0; i < weight_num; i++) {
    ctx->weights_requires_grad.at(i) = inputs.at(i + 1)->requires_grad();              // NOLINT
    ctx->biases_requires_grad.at(i) = inputs.at(i + 1 + weight_num)->requires_grad();  // NOLINT
  }

  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // x. idx_sum:1
  for (int32_t i = 0; i < weight_num; i++) {
    ctx->SaveTensorForBackward(JUST(VectorAt(inputs, i + 1)));  // weights. idx_sum:1+w
  }

  ctx->SaveTensorForBackward(JUST(VectorAt(outputs, 0)));  // final layers output. idx_sum:2+w
  for (int32_t i = 0; i < weight_num; i++) {
    ctx->SaveTensorForBackward(
        JUST(VectorAt(outputs, i + 1)));  // cublas aux. need minus 1. idx_sum:2+2w
  }
  for (int32_t i = 0; i < weight_num; i++) {
    ctx->SaveTensorForBackward(JUST(VectorAt(outputs, i + 1 + weight_num)));  // hidden.
  }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->skip_final_activation = JUST(composed_attrs.GetAttr<bool>("skip_final_activation"));

  return Maybe<void>::Ok();
}

Maybe<void> CublasFusedMLP::Apply(const CublasFusedMLPCaptureState* ctx,
                                  const TensorTuple& out_grads, TensorTuple* in_grads) const {
  int32_t weight_num = ctx->weight_num;
  in_grads->resize(1 + 2 * weight_num);
  std::shared_ptr<one::Tensor> last_bias_dy = JUST(VectorAt(out_grads, 0));

  if (!ctx->skip_final_activation) {
    // step1: use dy and final output to get last layer's relu grad.
    last_bias_dy = JUST(functional::ReluGrad(JUST(VectorAt(out_grads, 0)),
                                             JUST(VectorAt(ctx->SavedTensors(), 1 + weight_num))));
  }

  TensorTuple hiddens(weight_num);
  TensorTuple weights(weight_num);
  TensorTuple cublas_auxs(weight_num);
  TensorTuple dgrad(weight_num);

  std::shared_ptr<one::Tensor> x = JUST(VectorAt(ctx->SavedTensors(), 0));

  for (int32_t i = 0; i < weight_num; ++i) {
    weights[i] = JUST(VectorAt(ctx->SavedTensors(), 1 + i));
  }

  for (int32_t i = 0; i < weight_num; ++i) {
    cublas_auxs[i] = JUST(VectorAt(ctx->SavedTensors(), i + 2 + weight_num));
  }

  for (int32_t i = 0; i < weight_num; ++i) {
    hiddens[i] = JUST(VectorAt(ctx->SavedTensors(), i + 2 + 2 * weight_num));
  }

  std::shared_ptr<one::Tensor> cublas_dy = last_bias_dy;

  // Use Fully Fused MLP Backward.
  if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD", false)) {
    const std::vector<float> alpha_list(weight_num - 1, 1.0);
    const auto& fused_mlp_grad =
        JUST(functional::FusedMLPGrad(cublas_dy, JUST(VectorAt(ctx->SavedTensors(), 0)), weights,
                                      cublas_auxs, hiddens, alpha_list));
    if (ctx->x_requires_grad) {
      // dx:
      JUST(VectorAt(*in_grads, 0)) = fused_mlp_grad->at(0);
    }

    for (int32_t hidden_layer_idx = weight_num - 1; hidden_layer_idx > -1; hidden_layer_idx--) {
      if (JUST(VectorAt(ctx->biases_requires_grad, (hidden_layer_idx)))) {
        // dbias
        JUST(VectorAt(*in_grads, weight_num + hidden_layer_idx + 1)) =
            fused_mlp_grad->at(1 + hidden_layer_idx);  // NOLINT
      }

      // dw
      if (JUST(VectorAt(ctx->weights_requires_grad, hidden_layer_idx))) {
        JUST(VectorAt(*in_grads, (1 + hidden_layer_idx))) =
            fused_mlp_grad->at(1 + weight_num + hidden_layer_idx);
      }
    }
  } else {
    // step2: use reduce_sum to get last layer's bias grad.
    std::vector<int32_t> reduce_axes_vec{0};
    if (JUST(VectorAt(ctx->biases_requires_grad, weight_num - 1))) {
      JUST(VectorAt(*in_grads, 2 * weight_num)) =
          JUST(functional::ReduceSum(last_bias_dy, reduce_axes_vec, false));
    }

    for (int32_t hidden_layer_idx = weight_num - 1; hidden_layer_idx > 0; hidden_layer_idx--) {
      // If it is final layer, we use out_grads[0] as dy.
      if (hidden_layer_idx != weight_num - 1) {
        cublas_dy = JUST(VectorAt(dgrad, hidden_layer_idx + 1));
      }
      /*
      Here we use cublas to compute bias + relu + matmul grad.
      Then use Matmul to compute weight grad.
      */
      const auto& matmul_relu_bias_bgrad = JUST(functional::CublasBiasAddReluMatmulGrad(
          cublas_dy, JUST(VectorAt(weights, hidden_layer_idx)),
          JUST(VectorAt(cublas_auxs, hidden_layer_idx - 1)), /*alpha=*/1.0));

      // dgrad
      dgrad.at(hidden_layer_idx) = matmul_relu_bias_bgrad->at(0);  // NOLINT

      if (JUST(VectorAt(ctx->biases_requires_grad, (hidden_layer_idx - 1)))) {
        // dbias
        JUST(VectorAt(*in_grads, weight_num + hidden_layer_idx)) =
            matmul_relu_bias_bgrad->at(1);  // NOLINT
      }
      // dw
      if (JUST(VectorAt(ctx->weights_requires_grad, hidden_layer_idx))) {
        JUST(VectorAt(*in_grads, (1 + hidden_layer_idx))) = JUST(functional::MatMul(
            cublas_dy, JUST(VectorAt(hiddens, hidden_layer_idx - 1)), true, false, 1.0));
      }
    }

    // For the first layer, we need to use 2 matmul to get grads.
    std::shared_ptr<one::Tensor> last_dy;
    if (weight_num != 1) {
      last_dy = JUST(VectorAt(dgrad, 1));
    } else {
      last_dy = last_bias_dy;
    }

    if (ctx->x_requires_grad) {
      // dx:
      JUST(VectorAt(*in_grads, 0)) =
          JUST(functional::MatMul(last_dy, JUST(VectorAt(weights, 0)), false, false, 1.0));
    }
    if (JUST(VectorAt(ctx->weights_requires_grad, 0))) {
      // dw:
      JUST(VectorAt(*in_grads, 1)) = JUST(
          functional::MatMul(last_dy, JUST(VectorAt(ctx->SavedTensors(), 0)), true, false, 1.0));
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("cublas_fused_mlp", CublasFusedMLP);

}  // namespace one

}  // namespace oneflow
#endif  // CUDA_VERSION >= 11060
