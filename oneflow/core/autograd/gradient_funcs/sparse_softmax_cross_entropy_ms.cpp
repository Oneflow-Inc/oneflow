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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SparseSoftmaxCrossEntropyMsCaptureState : public AutoGradCaptureState {
  int64_t depth = 0;
};

class SparseSoftmaxCrossEntropyMs
    : public OpExprGradFunction<SparseSoftmaxCrossEntropyMsCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SparseSoftmaxCrossEntropyMsCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SparseSoftmaxCrossEntropyMsCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> SparseSoftmaxCrossEntropyMs::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropyMs::Capture(SparseSoftmaxCrossEntropyMsCaptureState* ctx,
                                                 const TensorTuple& inputs,
                                                 const TensorTuple& outputs,
                                                 const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->depth = JUST(composed_attrs.GetAttr<int64_t>("depth"));
  CHECK_EQ_OR_RETURN(inputs.size(), 2);                    // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 2);                   // NOLINT(maybe-need-error-msg)
  ctx->SaveTensorForBackward(JUST(VectorAt(outputs, 0)));  // prob
  ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));   // label
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropyMs::Apply(const SparseSoftmaxCrossEntropyMsCaptureState* ctx,
                                               const TensorTuple& out_grads,
                                               TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
  const auto& dy = JUST(VectorAt(out_grads, 1));
  const auto& prob = JUST(VectorAt(ctx->SavedTensors(), 0));
  const auto& label = JUST(VectorAt(ctx->SavedTensors(), 1));
  // SparseSoftmaxCrossEntropy has 2 inputs (prediction and label), and the second input does not
  // require gradient.
  in_grads->resize(2);
  JUST(VectorAt(*in_grads, 0)) =
      JUST(functional::SparseSoftmaxCrossEntropyMsGrad(dy, prob, label, ctx->depth));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_softmax_cross_entropy_ms", SparseSoftmaxCrossEntropyMs);

}  // namespace one
}  // namespace oneflow
