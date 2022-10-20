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

struct OneEmbeddingFusedLookupCaptureState : public AutoGradCaptureState {
  bool requires_grad{};
  std::string embedding_name{};
  int64_t line_size{};
  int64_t embedding_size{};
  int shadow_index{};
  int ids_index{};
  int input_num{};
};

class OneEmbeddingFusedLookup : public OpExprGradFunction<OneEmbeddingFusedLookupCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(OneEmbeddingFusedLookupCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_GE_OR_RETURN(inputs.size(), 2);                          // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();            // shadow
    ctx->shadow_index = ctx->SaveTensorForBackward(inputs.at(0));  // shadow
    ctx->ids_index = ctx->SaveTensorForBackward(inputs.at(1));     // id
    ctx->embedding_name = JUST(attrs.GetAttr<std::string>("embedding_name"));
    ctx->line_size = JUST(attrs.GetAttr<int64_t>("line_size"));
    ctx->embedding_size = JUST(attrs.GetAttr<int64_t>("embedding_size"));
    ctx->input_num = inputs.size();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OneEmbeddingFusedLookupCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(ctx->input_num);
    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    if (ctx->requires_grad) {
      JUST(functional::OneEmbeddingFusedLookupGrad(
          saved_tensors.at(ctx->ids_index), JUST(VectorAt(out_grads, 0)), ctx->embedding_name,
          ctx->line_size, ctx->embedding_size));
      (*in_grads)[0] = JUST(functional::ZerosLike(saved_tensors.at(ctx->shadow_index)));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("one_embedding_fused_lookup", OneEmbeddingFusedLookup);

}  // namespace one
}  // namespace oneflow
