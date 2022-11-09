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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SparseCrossEntropyCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  int64_t depth = -1;
  size_t prediction_index = -1;
  size_t label_index = -1;
};

template<bool is_distributed>
class SparseCrossEntropy : public OpExprGradFunction<SparseCrossEntropyCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SparseCrossEntropyCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->depth = JUST(composed_attrs.GetAttr<int64_t>("depth"));
    ctx->prediction_index = ctx->SaveTensorForBackward(inputs.at(0));  // prediction
    ctx->label_index = ctx->SaveTensorForBackward(inputs.at(1));       // label
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SparseCrossEntropyCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const auto& prediction = ctx->SavedTensors().at(ctx->prediction_index);
    const auto& label = ctx->SavedTensors().at(ctx->label_index);
    in_grads->resize(2);
    if (is_distributed) {
      in_grads->at(0) = JUST(
          functional::SparseCrossEntropyMsGrad(prediction, label, out_grads.at(0), ctx->depth));
    } else {
      in_grads->at(0) =
          JUST(functional::SparseCrossEntropyGrad(prediction, label, out_grads.at(0), ctx->depth));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_cross_entropy_ms", SparseCrossEntropy<true>);
REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_cross_entropy", SparseCrossEntropy<false>);

}  // namespace one
}  // namespace oneflow
