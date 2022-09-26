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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct SpmmCooCaptureState : public AutoGradCaptureState {
  bool requires_grad_b;
  size_t row_index = 0;
  size_t col_index = 0;
  size_t val_index = 0;
  int64_t a_num_rows = 0.0;
  int64_t a_num_cols = 0.0;
};

class SpmmCoo : public OpExprGradFunction<SpmmCooCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(SpmmCooCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const SpmmCooCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> SpmmCoo::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be null. ";
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> SpmmCoo::Capture(SpmmCooCaptureState* ctx, const TensorTuple& inputs,
                             const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 4);

  ctx->requires_grad_b = JUST(VectorAt(inputs, 3))->requires_grad();
  if (!ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->a_num_rows = JUST(composed_attrs.GetAttr<int64_t>("a_num_rows"));
  ctx->a_num_cols = JUST(composed_attrs.GetAttr<int64_t>("a_num_cols"));

  if (ctx->requires_grad_b) {
    ctx->row_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input a
    ctx->col_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // input a
    ctx->val_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 2)));  // input a
  }
  return Maybe<void>::Ok();
}

Maybe<void> SpmmCoo::Apply(const SpmmCooCaptureState* ctx, const TensorTuple& out_grads,
                           TensorTuple* in_grads) const {
  if (!ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "Out grad size should be equal to 1. ";

  in_grads->resize(4);

  if (ctx->requires_grad_b) {
    const auto& a_coo_row = JUST(VectorAt(ctx->SavedTensors(), ctx->row_index));
    const auto& a_coo_col = JUST(VectorAt(ctx->SavedTensors(), ctx->col_index));
    const auto& a_coo_val = JUST(VectorAt(ctx->SavedTensors(), ctx->val_index));

    JUST(VectorAt(*in_grads, 3)) =
        JUST(functional::SpmmCoo(a_coo_row, a_coo_col, a_coo_val, ctx->a_num_rows, ctx->a_num_cols,
                                 JUST(VectorAt(out_grads, 0))));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("spmm_coo", SpmmCoo);
}  // namespace one
}  // namespace oneflow
