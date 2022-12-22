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
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace one {

struct FusedMatmulBiasCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool weight_requires_grad = false;
  bool bias_requires_grad = false;
};

class FusedMatmulBias : public OpExprGradFunction<FusedMatmulBiasCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedMatmulBiasCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedMatmulBiasCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> FusedMatmulBias::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedMatmulBias::Capture(FusedMatmulBiasCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_GE_OR_RETURN(inputs.size(), 3)
      << "x, weight, and bias, [add_to_output] should all be included";
  ctx->x_requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
  ctx->weight_requires_grad = JUST(VectorAt(inputs, 1))->requires_grad();
  ctx->bias_requires_grad = JUST(VectorAt(inputs, 2))->requires_grad();

  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));

  return Maybe<void>::Ok();
}

Maybe<void> FusedMatmulBias::Apply(const FusedMatmulBiasCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "FusedMatmulBias more than one output";
  const auto& x = ctx->SavedTensors().at(0);
  const auto& weight = ctx->SavedTensors().at(1);

  if (ctx->x_requires_grad) {
    in_grads->at(0) =
        JUST(functional::MatMul(JUST(VectorAt(out_grads, 0)), weight, false, false, 1.0));
  }
  if (ctx->weight_requires_grad) {
    in_grads->at(1) = JUST(functional::BroadcastMatmulGradB(JUST(VectorAt(out_grads, 0)), x, 1.0));
  }
  if (ctx->bias_requires_grad) {
    const int64_t num_axes = out_grads.at(0)->shape()->NumAxes();
    std::vector<int32_t> reduce_axes_vec;
    reduce_axes_vec.reserve(num_axes - 1);
    for (int i = 0; i < num_axes - 1; i++) { reduce_axes_vec.push_back(i); }
    in_grads->at(2) =
        JUST(functional::ReduceSum(JUST(VectorAt(out_grads, 0)), reduce_axes_vec, false));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_matmul_bias", FusedMatmulBias);

}  // namespace one

}  // namespace oneflow
