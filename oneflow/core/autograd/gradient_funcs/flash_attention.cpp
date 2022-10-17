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
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct FlashAttentionCaptureState : public AutoGradCaptureState {
  bool query_requires_grad = false;
  bool key_requires_grad = false;
  bool value_requires_grad = false;
  int query_index = 0;
  int key_index = 1;
  int value_index = 2;
  int valid_seqlens_q_index = 3;
  int valid_seqlens_k_index = 4;
  int out_index = 5;
  int softmax_lse_index = 6;
  bool causal = false;
};

class FlashAttention : public OpExprGradFunction<FlashAttentionCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FlashAttentionCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 5);                         // NOLINT(maybe-need-error-msg)
    ctx->query_requires_grad = inputs.at(0)->requires_grad();     // query
    ctx->key_requires_grad = inputs.at(1)->requires_grad();       // key
    ctx->value_requires_grad = inputs.at(2)->requires_grad();     // value
    ctx->query_index = ctx->SaveTensorForBackward(inputs.at(0));  // query
    ctx->key_index = ctx->SaveTensorForBackward(inputs.at(1));    // key
    ctx->value_index = ctx->SaveTensorForBackward(inputs.at(2));  // value
    ctx->valid_seqlens_q_index = ctx->SaveTensorForBackward(inputs.at(3));  // valid_seqlens_q_index
    ctx->valid_seqlens_k_index = ctx->SaveTensorForBackward(inputs.at(4));  // valid_seqlens_k_index
    ctx->out_index = ctx->SaveTensorForBackward(outputs.at(0));             // out
    ctx->softmax_lse_index = ctx->SaveTensorForBackward(outputs.at(1));     // softmax_lse
    ctx->causal = JUST(attrs.GetAttr<bool>("causal"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FlashAttentionCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!(ctx->query_requires_grad || ctx->key_requires_grad || ctx->value_requires_grad)) {
      return Maybe<void>::Ok();
    }
    in_grads->resize(5);
    const auto& saved_tensors = ctx->SavedTensors();

    const auto& results = JUST(functional::FlashAttentionGrad(
        out_grads.at(0), saved_tensors.at(ctx->out_index), saved_tensors.at(ctx->softmax_lse_index),
        saved_tensors.at(ctx->query_index), saved_tensors.at(ctx->key_index),
        saved_tensors.at(ctx->value_index), saved_tensors.at(ctx->valid_seqlens_q_index),
        saved_tensors.at(ctx->valid_seqlens_k_index), ctx->causal));

    if (ctx->query_requires_grad) { (*in_grads)[0] = results->at(0); }
    if (ctx->key_requires_grad) { (*in_grads)[1] = results->at(1); }
    if (ctx->value_requires_grad) { (*in_grads)[2] = results->at(2); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("flash_attention", FlashAttention);

}  // namespace one
}  // namespace oneflow
