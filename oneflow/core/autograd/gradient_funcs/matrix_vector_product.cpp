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

struct MatrixVectorProductCaptureState : public AutoGradCaptureState {
  bool requires_grad_a = false;
  bool requires_grad_b = false;
  size_t a_index = 0;
  size_t b_index = 1;
};

class MatrixVectorProduct : public OpExprGradFunction<MatrixVectorProductCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MatrixVectorProductCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MatrixVectorProductCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> MatrixVectorProduct::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be null. ";
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> MatrixVectorProduct::Capture(MatrixVectorProductCaptureState* ctx,
                                         const TensorTuple& inputs, const TensorTuple& outputs,
                                         const AttrMap& attrs) const {
  ctx->requires_grad_a = JUST(VectorAt(inputs, 0))->requires_grad();
  ctx->requires_grad_b = JUST(VectorAt(inputs, 1))->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  if (ctx->requires_grad_a) {
    ctx->b_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // input b
  }
  if (ctx->requires_grad_b) {
    ctx->a_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input a
  }
  return Maybe<void>::Ok();
}

Maybe<void> MatrixVectorProduct::Apply(const MatrixVectorProductCaptureState* ctx,
                                       const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "Out grad size should be equal to 1. ";

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = JUST(VectorAt(ctx->SavedTensors(), ctx->b_index));
    JUST(VectorAt(*in_grads, 0)) =
        JUST(functional::MatrixVectorProductGradA(JUST(VectorAt(out_grads, 0)), input_b));
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = JUST(VectorAt(ctx->SavedTensors(), ctx->a_index));
    JUST(VectorAt(*in_grads, 1)) =
        JUST(functional::MatrixVectorProductGradB(JUST(VectorAt(out_grads, 0)), input_a));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matrix_vector_product", MatrixVectorProduct);

}  // namespace one
}  // namespace oneflow
