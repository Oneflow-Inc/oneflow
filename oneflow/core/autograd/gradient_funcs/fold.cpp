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

struct FoldInterpState : public AutoGradCaptureState {
  bool requires_grad = true;
  std::string data_format = "channels_first";
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> dilation_rate;
  std::vector<int32_t> padding;
  std::vector<int32_t> strides;
};

class Fold : public OpExprGradFunction<FoldInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FoldInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const FoldInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Fold::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Fold::Capture(FoldInterpState* ctx, const TensorTuple& inputs,
                          const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->dilation_rate = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation_rate"));
  ctx->padding = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  return Maybe<void>::Ok();
}

Maybe<void> Fold::Apply(const FoldInterpState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Unfold(out_grads.at(0), ctx->kernel_size, ctx->dilation_rate,
                                            ctx->padding, ctx->strides, ctx->data_format));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fold", Fold);

}  // namespace one
}  // namespace oneflow
