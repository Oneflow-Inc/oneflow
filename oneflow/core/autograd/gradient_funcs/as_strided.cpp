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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct AsStridedCaptureState : public AutoGradCaptureState {
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;
  bool requires_grad = false;
};

class AsStrided : public OpExprGradFunction<AsStridedCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(AsStridedCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const AsStridedCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> AsStrided::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> AsStrided::Capture(AsStridedCaptureState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("size"));
  ctx->stride = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stride"));
  ctx->storage_offset = JUST(composed_attrs.GetAttr<int64_t>("storage_offset"));
  return Maybe<void>::Ok();
}

Maybe<void> AsStrided::Apply(const AsStridedCaptureState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  const auto& input = ctx->SavedTensors().at(0);
  std::vector<int64_t> size = ctx->size;
  std::vector<int64_t> stride = ctx->stride;
  int64_t storage_offset = ctx->storage_offset;

  in_grads->at(0) =
      JUST(functional::AsStridedGrad(out_grads.at(0), input, size, stride, storage_offset));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("as_strided", AsStrided);

}  // namespace one
}  // namespace oneflow