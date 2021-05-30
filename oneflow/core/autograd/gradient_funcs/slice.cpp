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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

struct SliceOpExprInterpState : public OpExprInterpState {
  bool requires_grad;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class Slice : public OpExprGradFunction<SliceOpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    const std::string& op_name = fw_op_expr->op_name();
    std::vector<int64_t> start, stop, step;
    grad_op_ = JUST(op_expr_helper::SliceGradOp(start, stop, step, GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SliceOpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceOpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& like = ctx->SavedTensors().at(0);
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", ctx->start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", ctx->stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", ctx->step));

    in_grads->resize(1);
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0), like}, attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("slice", Slice);
REGISTER_OP_EXPR_GRAD_FUNCTION("slice_update", Slice);

}  // namespace one
}  // namespace oneflow
