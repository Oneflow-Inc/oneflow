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

#include <cstdint>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct FusedRowAttentionWithPairBiasCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = true;
  bool mask_requires_grad = false;
  float scale = 1.0;
  int64_t stride = 1;
};

class FusedRowAttentionWithPairBias
    : public OpExprGradFunction<FusedRowAttentionWithPairBiasCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedRowAttentionWithPairBiasCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedRowAttentionWithPairBiasCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedRowAttentionWithPairBias::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedRowAttentionWithPairBias::Capture(FusedRowAttentionWithPairBiasCaptureState* ctx,
                                                   const TensorTuple& inputs,
                                                   const TensorTuple& outputs,
                                                   const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // input, mask, pair
  // ctx->mask_requires_grad = inputs.at(1)->requires_grad(); // set mask.requires_grad=False
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->scale = JUST(composed_attrs.GetAttr<float>("scale"));
  auto shape = inputs.at(0)->shape();
  const int64_t h = shape->At(1), S = shape->At(2);
  ctx->stride = h * S;

  ctx->SaveTensorForBackward(outputs.at(0));  // y
  return Maybe<void>::Ok();
}

Maybe<void> FusedRowAttentionWithPairBias::Apply(
    const FusedRowAttentionWithPairBiasCaptureState* ctx, const TensorTuple& out_grads,
    TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // dy
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(3);  // input, mask, dropout_mask

  const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::TensorTuple>& input_grads = JUST(
      functional::FusedRowAttentionWithPairBiasGrad(y, out_grads.at(0), ctx->scale, ctx->stride));

  in_grads->at(0) = input_grads->at(0);  // input
  in_grads->at(2) = JUST(functional::ReduceSum(input_grads->at(1), std::vector<int32_t>{0},
                                               true));  // pair_grad: B,h,S,S -> 1, h, S, S

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_row_attention_with_pair_bias", FusedRowAttentionWithPairBias);

}  // namespace one
}  // namespace oneflow
