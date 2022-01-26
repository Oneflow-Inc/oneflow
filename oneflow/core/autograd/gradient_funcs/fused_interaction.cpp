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

namespace oneflow {
namespace one {

struct FusedInteractionCaptureState : public AutoGradCaptureState {
  bool dense_feature_requires_grad;
  bool sparse_feature_requires_grad;
  int64_t embedding_size;
  int64_t num_columns;
};

class FusedInteraction : public OpExprGradFunction<FusedInteractionCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedInteractionCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedInteractionCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedInteraction::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  return Maybe<void>::Ok();
}

Maybe<void> FusedInteraction::Capture(FusedInteractionCaptureState* ctx, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->dense_feature_requires_grad = inputs.at(0)->requires_grad();
  ctx->sparse_feature_requires_grad = inputs.at(1)->requires_grad();
  if (!ctx->dense_feature_requires_grad && !ctx->sparse_feature_requires_grad) {
    return Maybe<void>::Ok();
  }

  ctx->SaveTensorForBackward(outputs.at(1));  // concat_out
  const auto& sparse_feature_shape = inputs.at(1)->shape();
  ctx->embedding_size = sparse_feature_shape->At(2);
  ctx->num_columns = sparse_feature_shape->At(1);
  return Maybe<void>::Ok();
}

Maybe<void> FusedInteraction::Apply(const FusedInteractionCaptureState* ctx,
                                    const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->dense_feature_requires_grad && !ctx->sparse_feature_requires_grad) {
    return Maybe<void>::Ok();
  }
  in_grads->resize(2);
  // CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& concat_out = ctx->SavedTensors().at(0);
  const auto& grads = JUST(functional::FusedInteractionGrad(out_grads.at(0), concat_out,
                                                            ctx->embedding_size, ctx->num_columns));
  if (ctx->dense_feature_requires_grad) { in_grads->at(0) = grads->at(0); }
  if (ctx->sparse_feature_requires_grad) { in_grads->at(1) = grads->at(1); }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_interaction", FusedInteraction);

}  // namespace one
}  // namespace oneflow
