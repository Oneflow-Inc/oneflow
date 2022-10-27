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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct MaxUnpoolCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  size_t input_index = 0;
  size_t indices_index = 0;
  std::vector<int32_t> kernel_size;
};

class MaxUnpoolNdGrad : public OpExprGradFunction<MaxUnpoolCaptureState> {
 public:
  virtual ~MaxUnpoolNdGrad() = default;

  using OpExprGradFunction<MaxUnpoolCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MaxUnpoolCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MaxUnpoolCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> MaxUnpoolNdGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> MaxUnpoolNdGrad::Capture(MaxUnpoolCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->input_index = ctx->SaveTensorForBackward(inputs.at(0));
  ctx->indices_index = ctx->SaveTensorForBackward(inputs.at(1));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  return Maybe<void>::Ok();
}

Maybe<void> MaxUnpoolNdGrad::Apply(const MaxUnpoolCaptureState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_LE_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)

  int32_t ndims = ctx->kernel_size.size();
  const auto& input = ctx->SavedTensors().at(ctx->input_index);
  const auto& indices = ctx->SavedTensors().at(ctx->indices_index);

  in_grads->resize(2);
  (*in_grads)[0] = JUST(functional::MaxUnpoolNdGrad(input, indices, out_grads[0], ndims));

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_EXPR_GRAD_FUNCTION("max_unpool_1d", MaxUnpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_unpool_2d", MaxUnpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_unpool_3d", MaxUnpoolNdGrad);

}  // namespace one
}  // namespace oneflow
