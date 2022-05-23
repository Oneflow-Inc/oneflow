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
#if CUDA_VERSION >= 11040

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
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> CublasFusedMLP::Capture(CublasFusedMLPCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() % 2 == 1) << "Both weight and bias should be passed together. ";
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
  for (int32_t i = 0; i < weight_num - 1; i++) {
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

  TensorTuple hiddens(weight_num - 1);
  TensorTuple weights(weight_num);
  TensorTuple cublas_auxs(weight_num);
  TensorTuple dgrad(weight_num);

  std::shared_ptr<one::Tensor> x = JUST(VectorAt(ctx->SavedTensors(), 0));
  std::shared_ptr<one::Tensor> out = JUST(VectorAt(ctx->SavedTensors(), 1 + weight_num));

  for (int32_t i = 0; i < weight_num; ++i) {
    weights[i] = JUST(VectorAt(ctx->SavedTensors(), 1 + i));
  }

  for (int32_t i = 0; i < weight_num; ++i) {
    cublas_auxs[i] = JUST(VectorAt(ctx->SavedTensors(), i + 2 + weight_num));
  }

  for (int32_t i = 0; i < weight_num - 1; ++i) {
    hiddens[i] = JUST(VectorAt(ctx->SavedTensors(), i + 2 + 2 * weight_num));
  }

  std::shared_ptr<one::Tensor> last_bias_dy = JUST(VectorAt(out_grads, 0));

  if (!ctx->skip_final_activation) {
    // step1: use dy and final output to get last layer's relu grad.
    last_bias_dy = JUST(functional::ReluGrad(JUST(VectorAt(out_grads, 0)), out));
  }

  const bool last_layer_weight_requires_grad =
      JUST(VectorAt(ctx->weights_requires_grad, weight_num - 1));
  const bool last_layer_bias_requires_grad =
      JUST(VectorAt(ctx->biases_requires_grad, weight_num - 1));

  // For last layer, we use CublasMatmulBiasAddGrad to get wgrad and b grad.
  if ((last_layer_weight_requires_grad || last_layer_bias_requires_grad)) {
    // If there is only 1 layer, we use CublasMatmulBiasAddGrad to calculate first layer's dw.
    std::shared_ptr<one::Tensor> last_layer_x = x;
    if (weight_num != 1) { last_layer_x = JUST(VectorAt(hiddens, weight_num - 2)); }
    const auto& last_layer_wgrad_bgrad =
        JUST(functional::CublasMatmulBiasAddGrad(last_bias_dy, last_layer_x));
    if (last_layer_weight_requires_grad) {
      JUST(VectorAt(*in_grads, weight_num)) = JUST(VectorAt(*last_layer_wgrad_bgrad, 0));
    }
    if (last_layer_bias_requires_grad) {
      JUST(VectorAt(*in_grads, 2 * weight_num)) = JUST(VectorAt(*last_layer_wgrad_bgrad, 1));
    }
  }

  std::shared_ptr<one::Tensor> cublas_dy = last_bias_dy;
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
        JUST(VectorAt(cublas_auxs, hidden_layer_idx - 1))));

    // dgrad
    dgrad.at(hidden_layer_idx) = matmul_relu_bias_bgrad->at(0);  // NOLINT

    if (JUST(VectorAt(ctx->biases_requires_grad, (hidden_layer_idx - 1)))) {
      // dbias
      JUST(VectorAt(*in_grads, weight_num + hidden_layer_idx)) =
          matmul_relu_bias_bgrad->at(1);  // NOLINT
    }
    // dw, need to skip final layer, cause final layer's wgrad has used CublasMatmulBiasAddGrad to
    // calculate.
    if (JUST(VectorAt(ctx->weights_requires_grad, hidden_layer_idx))
        && hidden_layer_idx != weight_num - 1) {
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
  if (JUST(VectorAt(ctx->weights_requires_grad, 0)) && weight_num >= 2) {
    // If weight_num == 1, dw has been calculated by CublasMatmulBiasAddGrad, so we need to skip.
    // dw:
    JUST(VectorAt(*in_grads, 1)) =
        JUST(functional::MatMul(last_dy, JUST(VectorAt(ctx->SavedTensors(), 0)), true, false,
                                1.0));  // use x instead just vectorat
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("cublas_fused_mlp", CublasFusedMLP);

}  // namespace one

}  // namespace oneflow
#endif  // CUDA_VERSION >= 11040
