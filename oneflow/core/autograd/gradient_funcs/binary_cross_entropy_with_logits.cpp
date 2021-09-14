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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct BinaryCrossEntropyWithLogitsCaptureState : public AutoGradCaptureState {
  std::string reduction;
  bool has_pos_weight;
};

class BinaryCrossEntropyWithLogits
    : public OpExprGradFunction<BinaryCrossEntropyWithLogitsCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BinaryCrossEntropyWithLogitsCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BinaryCrossEntropyWithLogitsCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
  std::shared_ptr<OpExpr> grad_op_weight_;
  std::shared_ptr<OpExpr> grad_op_pos_;
  std::shared_ptr<OpExpr> grad_op_weight_pos_;
};

Maybe<void> BinaryCrossEntropyWithLogits::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  grad_op_ = JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad", GradientOpName(op_name))
                      .Input("input")
                      .Input("target")
                      .Input("dy")
                      .Output("dx")
                      .Build());
  grad_op_weight_ =
      JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad", GradientOpName(op_name))
               .Input("input")
               .Input("target")
               .Input("weight")
               .Input("dy")
               .Output("dx")
               .Build());
  grad_op_pos_ =
      JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad", GradientOpName(op_name))
               .Input("input")
               .Input("target")
               .Input("pos_weight")
               .Input("dy")
               .Output("dx")
               .Build());
  grad_op_weight_pos_ =
      JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad", GradientOpName(op_name))
               .Input("input")
               .Input("target")
               .Input("weight")
               .Input("pos_weight")
               .Input("dy")
               .Output("dx")
               .Build());
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogits::Capture(BinaryCrossEntropyWithLogitsCaptureState* ctx,
                                                  const TensorTuple& inputs,
                                                  const TensorTuple& outputs,
                                                  const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->reduction = JUST(composed_attrs.GetAttr<std::string>("reduction"));
  ctx->has_pos_weight = JUST(composed_attrs.GetAttr<bool>("has_pos_weight"));
  ctx->SaveTensorForBackward(inputs.at(0));  // input
  ctx->SaveTensorForBackward(inputs.at(1));  // target
  if (inputs.size() == 3) {
    ctx->SaveTensorForBackward(inputs.at(2));  // weight or pos_weight
  }
  if (inputs.size() == 4) {
    ctx->SaveTensorForBackward(inputs.at(2));  // weight
    ctx->SaveTensorForBackward(inputs.at(3));  // pos_weight
  }
  return Maybe<void>::Ok();
}
Maybe<void> BinaryCrossEntropyWithLogits::Apply(const BinaryCrossEntropyWithLogitsCaptureState* ctx,
                                                const TensorTuple& out_grads,
                                                TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& dy = out_grads.at(0);
  const auto& input = ctx->SavedTensors().at(0);
  const auto& target = ctx->SavedTensors().at(1);

  MutableAttrMap attrs;
  JUST(attrs.SetAttr<std::string>("reduction", ctx->reduction));
  JUST(attrs.SetAttr<bool>("has_pos_weight", ctx->has_pos_weight));

  in_grads->resize(ctx->SavedTensors().size());
  if (ctx->SavedTensors().size() == 3) {
    if (ctx->has_pos_weight) {
      const auto& pos_weight = ctx->SavedTensors().at(2);
      in_grads->at(0) = JUST(
          OpInterpUtil::Dispatch<Tensor>(*grad_op_pos_, {input, target, pos_weight, dy}, attrs));
    } else {
      const auto& weight = ctx->SavedTensors().at(2);
      in_grads->at(0) = JUST(
          OpInterpUtil::Dispatch<Tensor>(*grad_op_weight_, {input, target, weight, dy}, attrs));
    }
  } else if (ctx->SavedTensors().size() == 4) {
    const auto& weight = ctx->SavedTensors().at(2);
    const auto& pos_weight = ctx->SavedTensors().at(3);
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(
        *grad_op_weight_pos_, {input, target, weight, pos_weight, dy}, attrs));
  } else {
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {input, target, dy}, attrs));
  }
  return Maybe<void>::Ok();
}
REGISTER_OP_EXPR_GRAD_FUNCTION("binary_cross_entropy_with_logits", BinaryCrossEntropyWithLogits);
}  // namespace one
}  // namespace oneflow
