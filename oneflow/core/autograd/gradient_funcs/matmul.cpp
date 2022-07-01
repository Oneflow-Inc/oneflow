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

namespace oneflow {
namespace one {

struct MatmulCaptureState : public AutoGradCaptureState {
  bool transpose_a;
  bool transpose_b;
  double alpha;
  bool requires_grad_a;
  bool requires_grad_b;
  size_t a_index;
  size_t b_index;
};

class Matmul : public OpExprGradFunction<MatmulCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> Matmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_a = inputs.at(0)->requires_grad();
  ctx->requires_grad_b = inputs.at(1)->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->transpose_a = JUST(composed_attrs.GetAttr<bool>("transpose_a"));
  ctx->transpose_b = JUST(composed_attrs.GetAttr<bool>("transpose_b"));
  ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
  if (ctx->requires_grad_a) {
    ctx->b_index = ctx->SaveTensorForBackward(inputs.at(1));  // input b
  }
  if (ctx->requires_grad_b) {
    ctx->a_index = ctx->SaveTensorForBackward(inputs.at(0));  // input a
  }
  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(functional::MatMul(input_b, out_grads.at(0), ctx->transpose_b, true, ctx->alpha));
    } else {
      in_grads->at(0) = JUST(
          functional::MatMul(out_grads.at(0), input_b, false, !(ctx->transpose_b), ctx->alpha));
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(functional::MatMul(out_grads.at(0), input_a, true, ctx->transpose_a, ctx->alpha));
    } else {
      in_grads->at(1) = JUST(
          functional::MatMul(input_a, out_grads.at(0), !(ctx->transpose_a), false, ctx->alpha));
    }
  }

  return Maybe<void>::Ok();
}

class BroadcastMatmul : public Matmul {
 public:
  Maybe<void> Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BroadcastMatmul::Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(functional::MatMul(input_b, out_grads.at(0), ctx->transpose_b, true, ctx->alpha));
    } else {
      in_grads->at(0) = JUST(
          functional::MatMul(out_grads.at(0), input_b, false, !(ctx->transpose_b), ctx->alpha));
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(functional::BroadcastMatmulGradB(out_grads.at(0), input_a, ctx->alpha));
    } else {
      in_grads->at(1) =
          JUST(functional::BroadcastMatmulGradB(input_a, out_grads.at(0), ctx->alpha));
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("batch_matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_matmul", BroadcastMatmul);

}  // namespace one
}  // namespace oneflow
