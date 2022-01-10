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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct EmbeddingCaptureState : public AutoGradCaptureState {
  int32_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool requires_grad = false;
};

class Embedding : public OpExprGradFunction<EmbeddingCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(EmbeddingCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const EmbeddingCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override ;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Embedding::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Embedding::Capture(EmbeddingCaptureState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->padding_idx = JUST(composed_attrs.GetAttr<int32_t>("padding_idx"));
  ctx->scale_grad_by_freq = JUST(composed_attrs.GetAttr<bool>("scale_grad_by_freq"));
  return Maybe<void>::Ok();
}

Maybe<void> Embedding::Apply(const EmbeddingCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const auto& weight = ctx->SavedTensors().at(0);
  const auto& indices = ctx->SavedTensors().at(1);
  int32_t padding_idx = ctx->padding_idx;
  bool scale_grad_by_freq = ctx->scale_grad_by_freq;

  in_grads->at(0) =
      JUST(functional::EmbeddingGrad(out_grads.at(0), weight, indices, padding_idx, scale_grad_by_freq));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("embedding", Embedding);

}  // namespace one
}  // namespace oneflow