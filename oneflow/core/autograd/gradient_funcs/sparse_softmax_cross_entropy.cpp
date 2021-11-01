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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct SparseSoftmaxCrossEntropyCaptureState : public AutoGradCaptureState {
  int64_t depth;
};

class SparseSoftmaxCrossEntropy : public OpExprGradFunction<SparseSoftmaxCrossEntropyCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SparseSoftmaxCrossEntropyCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SparseSoftmaxCrossEntropyCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> SparseSoftmaxCrossEntropy::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  grad_op_ =
      JUST(op_expr_helper::SparseSoftmaxCrossEntropyGradOp(/*depth=*/-1, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropy::Capture(SparseSoftmaxCrossEntropyCaptureState* ctx,
                                               const TensorTuple& inputs,
                                               const TensorTuple& outputs,
                                               const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->depth = JUST(composed_attrs.GetAttr<int64_t>("depth"));
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  CHECK_EQ_OR_RETURN(outputs.size(), 2);
  ctx->SaveTensorForBackward(outputs.at(0));  // prob
  ctx->SaveTensorForBackward(inputs.at(1));   // label
  return Maybe<void>::Ok();
}

Maybe<void> SparseSoftmaxCrossEntropy::Apply(const SparseSoftmaxCrossEntropyCaptureState* ctx,
                                             const TensorTuple& out_grads,
                                             TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 2);
  const auto& dy = out_grads.at(1);
  const auto& prob = ctx->SavedTensors().at(0);
  const auto& label = ctx->SavedTensors().at(1);
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<int64_t>("depth", ctx->depth));
  // SparseSoftmaxCrossEntropy has 2 inputs (prediction and label), and the second input does not
  // require gradient.
  in_grads->resize(2);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {prob, label, dy}, attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("sparse_softmax_cross_entropy", SparseSoftmaxCrossEntropy);

}  // namespace one
}  // namespace oneflow
