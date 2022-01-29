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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {

namespace one {

struct FusedMatmulBiasAddReluCaptureState : public AutoGradCaptureState {
  bool transpose_a;
  bool transpose_b;
  double alpha;
  bool requires_grad_a;
  bool requires_grad_b;
  bool requires_grad_bias;
  size_t a_index;
  size_t b_index;
  size_t bias_index;
  size_t out_index;
};

class FusedMatmulBiasAddRelu : public OpExprGradFunction<FusedMatmulBiasAddReluCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedMatmulBiasAddReluCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedMatmulBiasAddReluCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> FusedMatmulBiasAddRelu::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedMatmulBiasAddRelu::Capture(FusedMatmulBiasAddReluCaptureState* ctx,
                                            const TensorTuple& inputs, const TensorTuple& outputs,
                                            const AttrMap& attrs) const {
  ctx->requires_grad_a = inputs.at(0)->requires_grad();
  ctx->requires_grad_b = inputs.at(1)->requires_grad();
  ctx->requires_grad_bias = inputs.at(2)->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b && !ctx->requires_grad_bias) {
    return Maybe<void>::Ok();
  }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->transpose_a = JUST(composed_attrs.GetAttr<bool>("transpose_a"));
  ctx->transpose_b = JUST(composed_attrs.GetAttr<bool>("transpose_b"));
  ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));

  if (ctx->requires_grad_a) {
    ctx->b_index = ctx->SaveTensorForBackward(inputs.at(1));  // input b
  }
  if (ctx->requires_grad_b) {
    ctx->a_index = ctx->SaveTensorForBackward(inputs.at(0));  // input a
  }
  ctx->out_index = ctx->SaveTensorForBackward(outputs.at(0));  // output
  return Maybe<void>::Ok();
}

Maybe<void> FusedMatmulBiasAddRelu::Apply(const FusedMatmulBiasAddReluCaptureState* ctx,
                                          const TensorTuple& out_grads,
                                          TensorTuple* in_grads) const {
  in_grads->resize(3);
  std::shared_ptr<one::Tensor> relu_grad =
      JUST(functional::ReluGrad(out_grads.at(0), ctx->SavedTensors().at(ctx->out_index)));
  if (ctx->requires_grad_bias) {
    // TODO: Currently Only support 2d fused_matmul.
    // so here we hard encode bias reduce axis as 0.
    std::vector<int32_t> reduce_axes_vec{0};
    in_grads->at(2) = JUST(functional::ReduceSum(relu_grad, reduce_axes_vec, false));
  }

  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(functional::MatMul(input_b, relu_grad, ctx->transpose_b, true, ctx->alpha));
    } else {
      in_grads->at(0) =
          JUST(functional::MatMul(relu_grad, input_b, false, !(ctx->transpose_b), ctx->alpha));
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(functional::MatMul(relu_grad, input_a, true, ctx->transpose_a, ctx->alpha));
    } else {
      in_grads->at(1) =
          JUST(functional::MatMul(input_a, relu_grad, !(ctx->transpose_a), false, ctx->alpha));
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_matmul_bias_add_relu", FusedMatmulBiasAddRelu);

}  // namespace one

}  // namespace oneflow