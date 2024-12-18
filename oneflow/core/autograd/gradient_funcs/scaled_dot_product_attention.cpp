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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"
#if CUDA_VERSION >= 11070

namespace oneflow {

namespace one {

struct ScaledDotProductFlashAttentionCaptureState : public AutoGradCaptureState {
  bool query_requires_grad = true;
  bool key_requires_grad = true;
  bool value_requires_grad = true;
  size_t query_idx = 0;
  size_t key_idx = 0;
  size_t value_idx = 0;
  size_t out_idx = 0;
  size_t softmax_lse_idx = 0;
  size_t rng_state_idx = 0;
  float p_dropout = .0f;
  float softmax_scale = .0f;
  bool is_causal = false;
};

class ScaledDotProductFlashAttention
    : public OpExprGradFunction<ScaledDotProductFlashAttentionCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be None. ";
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScaledDotProductFlashAttentionCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 3) << "Input size should be equal to 3. ";
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->p_dropout = JUST(composed_attrs.GetAttr<float>("p_dropout"));
    ctx->softmax_scale = JUST(composed_attrs.GetAttr<float>("softmax_scale"));
    ctx->is_causal = JUST(composed_attrs.GetAttr<bool>("is_causal"));
    ctx->query_requires_grad = JUST(oneflow::VectorAt(inputs, 0))->requires_grad();
    ctx->key_requires_grad = JUST(oneflow::VectorAt(inputs, 1))->requires_grad();
    ctx->value_requires_grad = JUST(oneflow::VectorAt(inputs, 2))->requires_grad();
    ctx->query_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 0)));
    ctx->key_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 1)));
    ctx->value_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(inputs, 2)));
    ctx->out_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(outputs, 0)));
    ctx->softmax_lse_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(outputs, 1)));
    ctx->rng_state_idx = ctx->SaveTensorForBackward(JUST(oneflow::VectorAt(outputs, 2)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScaledDotProductFlashAttentionCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 3) << "Out grads size should be equal to 3. ";
    std::shared_ptr<oneflow::one::TensorTuple> grads;
    in_grads->resize(3);
    grads = JUST(functional::ScaledDotProductFlashAttentionGrad(
        JUST(oneflow::VectorAt(out_grads, 0)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->query_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->key_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->value_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->out_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->softmax_lse_idx)),
        JUST(oneflow::VectorAt(ctx->SavedTensors(), ctx->rng_state_idx)), ctx->p_dropout,
        ctx->is_causal, ctx->softmax_scale));

    if (ctx->query_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(oneflow::VectorAt(*grads, 0));
    }
    if (ctx->key_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 1)) = JUST(oneflow::VectorAt(*grads, 1));
    }
    if (ctx->value_requires_grad) {
      JUST(oneflow::VectorAt(*in_grads, 2)) = JUST(oneflow::VectorAt(*grads, 2));
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scaled_dot_product_flash_attention",
                               ScaledDotProductFlashAttention);

}  // namespace one

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11070
