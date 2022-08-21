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
  bool requires_grad;
  std::string key_value_store_options;
  int ids_index;
};

class OneEmbeddingLookup : public OpExprGradFunction<OneEmbeddingLookupCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(OneEmbeddingLookupCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();  // shadow
    LOG(ERROR) << "ctx->requires_grad " << ctx->requires_grad;
    ctx->ids_index = ctx->SaveTensorForBackward(inputs.at(1));  // id
    LOG(ERROR) << "ctx->ids_index " << ctx->ids_index;
    ctx->key_value_store_options = "";  // TODO
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OneEmbeddingLookupCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& saved_tensors = ctx->SavedTensors();
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    // in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] = JUST(functional::OneEmbeddingLookupGrad(
          saved_tensors.at(0), JUST(VectorAt(out_grads, 0)), ctx->key_value_store_options));
    }
    LOG(ERROR) << "Apply ";
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("embedding_lookup_placeholder", OneEmbeddingLookup);

}  // namespace one
}  // namespace oneflow
