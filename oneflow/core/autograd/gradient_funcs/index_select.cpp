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

struct IndexSelectCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  int32_t dim = 0;
};

class IndexSelect : public OpExprGradFunction<IndexSelectCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(IndexSelectCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const IndexSelectCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> IndexSelect::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> IndexSelect::Capture(IndexSelectCaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->dim = JUST(composed_attrs.GetAttr<int32_t>("dim"));
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> IndexSelect::Apply(const IndexSelectCaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  std::cout << "enter IndexSelect::Apply" << std::endl;
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  const auto& x = ctx->SavedTensors().at(0);
  const auto& index = ctx->SavedTensors().at(1);
  if (ctx->requires_grad) {
    (*in_grads)[0] = JUST(functional::IndexSelectGrad(out_grads[0], x, index, ctx->dim));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("index_select", IndexSelect);

}  // namespace one
}  // namespace oneflow
