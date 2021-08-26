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
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

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

class MatmulBase : public OpExprGradFunction<MatmulCaptureState> {
 public:
  Maybe<void> Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_a_op_;
  std::shared_ptr<OpExpr> grad_b_op_;
};

Maybe<void> MatmulBase::Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
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

Maybe<void> MatmulBase::Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  MutableAttrMap attrs_a;
  MutableAttrMap attrs_b;
  JUST(attrs_a.SetAttr<double>("alpha", ctx->alpha));
  JUST(attrs_b.SetAttr<double>("alpha", ctx->alpha));

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      JUST(attrs_a.SetAttr<bool>("transpose_a", ctx->transpose_b));
      JUST(attrs_a.SetAttr<bool>("transpose_b", true));
      in_grads->at(0) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_a_op_, {input_b, out_grads.at(0)}, attrs_a));
    } else {
      JUST(attrs_a.SetAttr<bool>("transpose_a", false));
      JUST(attrs_a.SetAttr<bool>("transpose_b", !(ctx->transpose_b)));
      in_grads->at(0) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_a_op_, {out_grads.at(0), input_b}, attrs_a));
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->transpose_b) {
      JUST(attrs_b.SetAttr<bool>("transpose_a", true));
      JUST(attrs_b.SetAttr<bool>("transpose_b", ctx->transpose_a));
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op_, {out_grads.at(0), input_a}, attrs_b));
    } else {
      JUST(attrs_b.SetAttr<bool>("transpose_a", !(ctx->transpose_a)));
      JUST(attrs_b.SetAttr<bool>("transpose_b", false));
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op_, {input_a, out_grads.at(0)}, attrs_b));
    }
  }

  return Maybe<void>::Ok();
}

class Matmul : public MatmulBase {
 public:
  Maybe<void> Init(const OpExpr& op) override;
};

Maybe<void> Matmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();

  grad_a_op_ = JUST(op_expr_helper::MatmulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                             /*alpha=*/1.0, GradientOpName(op_name + "_a")));
  grad_b_op_ = JUST(op_expr_helper::MatmulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                             /*alpha=*/1.0, GradientOpName(op_name + "_b")));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", Matmul);

class BatchMatmul : public MatmulBase {
 public:
  Maybe<void> Init(const OpExpr& op) override;
};

Maybe<void> BatchMatmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();

  grad_a_op_ = JUST(op_expr_helper::BatchMatmulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                                  /*alpha=*/1.0, GradientOpName(op_name + "_a")));
  grad_b_op_ = JUST(op_expr_helper::BatchMatmulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                                  /*alpha=*/1.0, GradientOpName(op_name + "_b")));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("batch_matmul", BatchMatmul);

class BroadcastMatmul : public MatmulBase {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BroadcastMatmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();

  grad_a_op_ =
      JUST(op_expr_helper::BroadcastMatmulOp(/*transpose_a=*/false, /*transpose_b=*/false,
                                             /*alpha=*/1.0, GradientOpName(op_name + "_a")));
  grad_b_op_ =
      JUST(op_expr_helper::BroadcastMatmulGradBOp(/*alpha=*/1.0, GradientOpName(op_name + "_b")));
  return Maybe<void>::Ok();
}

Maybe<void> BroadcastMatmul::Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  MutableAttrMap attrs_a;
  MutableAttrMap attrs_b;
  JUST(attrs_a.SetAttr<double>("alpha", ctx->alpha));
  JUST(attrs_a.SetAttr<bool>("transpose_a", ctx->transpose_a));
  JUST(attrs_a.SetAttr<bool>("transpose_b", !(ctx->transpose_b)));
  JUST(attrs_b.SetAttr<double>("alpha", ctx->alpha));

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*grad_a_op_, {out_grads.at(0), input_b}, attrs_a));
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (!ctx->transpose_b) {
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op_, {input_a, out_grads.at(0)}, attrs_b));
    } else {
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*grad_b_op_, {out_grads.at(0), input_a}, attrs_b));
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_matmul", BroadcastMatmul);

}  // namespace one
}  // namespace oneflow
