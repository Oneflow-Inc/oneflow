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
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct FusedDotFeatureInteractionCaptureState : public AutoGradCaptureState {
  bool need_grad_op = false;
  std::vector<bool> features_requires_grad;
  std::vector<int32_t> feature_dims;
  int32_t output_concat_grad_dim = 0;
  bool self_interaction = false;
  bool has_output_concat = false;
  bool has_output_concat_grad = false;
  std::string pooling;
};

class FusedDotFeatureInteraction
    : public OpExprGradFunction<FusedDotFeatureInteractionCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedDotFeatureInteractionCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedDotFeatureInteractionCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedDotFeatureInteraction::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  return Maybe<void>::Ok();
}

Maybe<void> FusedDotFeatureInteraction::Capture(FusedDotFeatureInteractionCaptureState* ctx,
                                                const TensorTuple& inputs,
                                                const TensorTuple& outputs,
                                                const AttrMap& attrs) const {
  ctx->has_output_concat = JUST(attrs.GetAttr<bool>("has_output_concat"));
  int32_t num_features = 0;
  if (ctx->has_output_concat) {
    num_features = inputs.size() - 1;
    const auto& output_concat = JUST(oneflow::VectorAt(inputs, num_features));
    ctx->has_output_concat_grad = output_concat->requires_grad();
    ctx->output_concat_grad_dim = output_concat->shape()->At(1);
  } else {
    num_features = inputs.size();
  }
  if (ctx->has_output_concat_grad) { ctx->need_grad_op = true; }
  ctx->features_requires_grad.resize(num_features);
  ctx->feature_dims.resize(num_features);
  for (int32_t i = 0; i < num_features; ++i) {
    const auto& feature = JUST(oneflow::VectorAt(inputs, i));
    ctx->features_requires_grad[i] = feature->requires_grad();
    ctx->feature_dims[i] = feature->shape()->At(1);
    if (feature->requires_grad()) { ctx->need_grad_op = true; }
    ctx->SaveTensorForBackward(feature);
  }
  ctx->pooling = JUST(attrs.GetAttr<std::string>("pooling"));
  if (!ctx->need_grad_op) { return Maybe<void>::Ok(); }
  ctx->self_interaction = JUST(attrs.GetAttr<bool>("self_interaction"));
  return Maybe<void>::Ok();
}

Maybe<void> FusedDotFeatureInteraction::Apply(const FusedDotFeatureInteractionCaptureState* ctx,
                                              const TensorTuple& out_grads,
                                              TensorTuple* in_grads) const {
  if (!ctx->need_grad_op) { return Maybe<void>::Ok(); }
  int32_t num_features = ctx->features_requires_grad.size();
  in_grads->resize(num_features + 1);
  TensorTuple features(num_features);
  for (int i = 0; i < num_features; ++i) {
    features[i] = JUST(oneflow::VectorAt(ctx->SavedTensors(), i));
  }
  std::shared_ptr<oneflow::one::TensorTuple> grads;
  grads = JUST(functional::FusedDotFeatureInteractionGrad(
      JUST(oneflow::VectorAt(out_grads, 0)), features, ctx->has_output_concat,
      ctx->self_interaction, ctx->output_concat_grad_dim, ctx->pooling));
  for (int32_t i = 0; i < num_features; ++i) {
    if (JUST(oneflow::VectorAt(ctx->features_requires_grad, i))) {
      JUST(oneflow::VectorAt(*in_grads, i)) = JUST(oneflow::VectorAt(*grads, i));
    }
  }
  if (ctx->has_output_concat_grad) {
    JUST(oneflow::VectorAt(*in_grads, num_features)) =
        JUST(oneflow::VectorAt(*grads, num_features));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_dot_feature_interaction", FusedDotFeatureInteraction);

}  // namespace one
}  // namespace oneflow
