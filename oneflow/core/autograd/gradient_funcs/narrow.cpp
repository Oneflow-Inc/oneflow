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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct NarrowOpInterpState : public OpExprInterpState {
  bool requires_grad;
  int64_t dim;
  int64_t start;
  int64_t length;
};

class NarrowOp : public OpExprGradFunction<NarrowOpInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(NarrowOpInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->dim = JUST(composed_attrs.GetAttr<int64_t>("dim"));
    ctx->start = JUST(composed_attrs.GetAttr<int64_t>("start"));
    ctx->length = JUST(composed_attrs.GetAttr<int64_t>("length"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NarrowOpInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (ctx->requires_grad) {
      const auto& like = ctx->SavedTensors().at(0);
      in_grads->resize(1);
      in_grads->at(0) =
          JUST(functional::NarrowGrad(out_grads.at(0), like, ctx->dim, ctx->start, ctx->length));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("narrow", NarrowOp);

}  // namespace one
}  // namespace oneflow
