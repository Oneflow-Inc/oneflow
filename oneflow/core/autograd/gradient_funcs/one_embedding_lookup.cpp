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

struct OneEmbeddingLookupCaptureState : public AutoGradCaptureState {
  bool requires_grad{};
  DataType dtype;
  std::string embedding_name{};
  int64_t line_size{};
  int64_t embedding_size{};
  bool is_full_cache{};
  int32_t num_tables{};
  std::string embedding_tables{};
  int64_t seed{};
  int shadow_index{};
  int ids_index{};
  bool has_table_ids;
  int table_ids_index{};
  int input_num{};
};

class OneEmbeddingLookup : public OpExprGradFunction<OneEmbeddingLookupCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(OneEmbeddingLookupCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->requires_grad = inputs.at(0)->requires_grad();            // shadow
    ctx->shadow_index = ctx->SaveTensorForBackward(inputs.at(0));  // shadow
    ctx->ids_index = ctx->SaveTensorForBackward(inputs.at(1));     // id
    if (inputs.size() == 3) {
      ctx->has_table_ids = true;
      ctx->table_ids_index = ctx->SaveTensorForBackward(inputs.at(2));
    } else {
      CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
      ctx->has_table_ids = false;
    }
    ctx->dtype = JUST(attrs.GetAttr<DataType>("dtype"));
    ctx->embedding_name = JUST(attrs.GetAttr<std::string>("embedding_name"));
    ctx->line_size = JUST(attrs.GetAttr<int64_t>("line_size"));
    ctx->embedding_size = JUST(attrs.GetAttr<int64_t>("embedding_size"));
    ctx->is_full_cache = JUST(attrs.GetAttr<bool>("is_full_cache"));
    ctx->num_tables = JUST(attrs.GetAttr<int32_t>("num_tables"));
    ctx->embedding_tables = JUST(attrs.GetAttr<std::string>("embedding_tables"));
    ctx->seed = JUST(attrs.GetAttr<int64_t>("seed"));
    ctx->input_num = inputs.size();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OneEmbeddingLookupCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(ctx->input_num);
    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    std::shared_ptr<Tensor> table_ids;
    if (ctx->has_table_ids) {
      table_ids = saved_tensors.at(ctx->table_ids_index);
    } else {
      table_ids = nullptr;
    }
    if (ctx->requires_grad) {
      JUST(functional::OneEmbeddingLookupGrad(
          saved_tensors.at(ctx->ids_index), table_ids, JUST(VectorAt(out_grads, 0)),
          DType(ctx->dtype), ctx->embedding_name, ctx->line_size, ctx->embedding_size,
          ctx->is_full_cache, ctx->num_tables, ctx->embedding_tables, ctx->seed));
      (*in_grads)[0] = JUST(functional::ZerosLike(saved_tensors.at(ctx->shadow_index)));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("embedding_lookup_placeholder", OneEmbeddingLookup);

}  // namespace one
}  // namespace oneflow
