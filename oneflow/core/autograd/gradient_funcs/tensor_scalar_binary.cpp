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

struct TensorScalarCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
  bool scalar_requires_grad;
};

class TensorScalarAddOrSub : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  TensorScalarAddOrSub() = default;
  virtual ~TensorScalarAddOrSub() = default;

  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;

 protected:
  std::shared_ptr<OpExpr> identity_op_;
  std::shared_ptr<OpExpr> reduce_sum_op_;
  // Only used by TensorScalarSub.
  std::shared_ptr<OpExpr> scalar_mul_op_;
};

Maybe<void> TensorScalarAddOrSub::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  identity_op_ = JUST(op_expr_helper::IdentityOp(GradientOpName(op_name + "_x")));
  reduce_sum_op_ = JUST(op_expr_helper::ReduceSumOp(
      /*axis=*/{-1}, /*keepdims=*/false, GradientOpName(op_name + "_scalar")));
  scalar_mul_op_ =
      JUST(op_expr_helper::ScalarMulOp<float>(-1.f, GradientOpName(op_name + "_scalar_mul")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarAddOrSub::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                          const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  return Maybe<void>::Ok();
}

class TensorScalarAdd : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*identity_op_, {out_grads.at(0)}));
    }
    if (ctx->scalar_requires_grad) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axes_vec));
      in_grads->at(1) =
          JUST(OpInterpUtil::Dispatch<Tensor>(*reduce_sum_op_, {out_grads.at(0)}, attrs));
    }
    return Maybe<void>::Ok();
  }
};

class TensorScalarSub : public TensorScalarAddOrSub {
 public:
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*identity_op_, {out_grads.at(0)}));
    }
    if (ctx->scalar_requires_grad) {
      int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
      std::vector<int32_t> axes_vec(num_axes);
      std::iota(axes_vec.begin(), axes_vec.end(), 0);
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axes_vec));
      const auto& reduce_sum =
          JUST(OpInterpUtil::Dispatch<Tensor>(*reduce_sum_op_, {out_grads.at(0)}, attrs));
      in_grads->at(1) = JUST(OpInterpUtil::Dispatch<Tensor>(*scalar_mul_op_, {reduce_sum}));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_add_by_tensor", TensorScalarAdd);
REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_sub_by_tensor", TensorScalarSub);

class TensorScalarMul : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> scalar_mul_op_;
  std::shared_ptr<OpExpr> multiply_op_;
  std::shared_ptr<OpExpr> reduce_sum_op_;
};

Maybe<void> TensorScalarMul::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  scalar_mul_op_ = JUST(op_expr_helper::ScalarMulByTensorOp(GradientOpName(op_name + "_x")));
  multiply_op_ = JUST(op_expr_helper::MultiplyOp(GradientOpName(op_name + "_scalar_mul")));
  reduce_sum_op_ = JUST(op_expr_helper::ReduceSumOp(
      /*axis=*/{-1}, /*keepdims=*/false, GradientOpName(op_name + "_scalar_reduce_sum")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs.at(1)); }
  if (ctx->scalar_requires_grad) { ctx->SaveTensorForBackward(inputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarMul::Apply(const TensorScalarCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (ctx->x_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*scalar_mul_op_, {out_grads.at(0), scalar}));
  }
  if (ctx->scalar_requires_grad) {
    const auto& x = ctx->SavedTensors().at(ctx->x_requires_grad);
    const auto& y = JUST(OpInterpUtil::Dispatch<Tensor>(*multiply_op_, {out_grads.at(0), x}));
    int32_t num_axes = out_grads.at(0)->shape()->NumAxes();
    std::vector<int32_t> axes_vec(num_axes);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axes_vec));
    in_grads->at(1) = JUST(OpInterpUtil::Dispatch<Tensor>(*reduce_sum_op_, {y}, attrs));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_mul_by_tensor", TensorScalarMul);

class TensorScalarDiv : public OpExprGradFunction<TensorScalarCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TensorScalarCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<OpExpr> tensor_scalar_div_op_;
  std::shared_ptr<OpExpr> broadcast_div_grad_op_;
};

Maybe<void> TensorScalarDiv::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  tensor_scalar_div_op_ = JUST(op_expr_helper::ScalarDivByTensorOp(GradientOpName(op_name + "_x")));
  broadcast_div_grad_op_ =
      JUST(op_expr_helper::BroadcastDivGradOp(GradientOpName(op_name + "_scalar")));
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Capture(TensorScalarCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->x_requires_grad = inputs.at(0)->requires_grad();
  ctx->scalar_requires_grad = inputs.at(1)->requires_grad();
  if (ctx->x_requires_grad || ctx->scalar_requires_grad) {
    ctx->SaveTensorForBackward(inputs.at(1));
  }
  if (ctx->scalar_requires_grad) { ctx->SaveTensorForBackward(outputs.at(0)); }
  return Maybe<void>::Ok();
}

Maybe<void> TensorScalarDiv::Apply(const TensorScalarCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (ctx->x_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    in_grads->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*tensor_scalar_div_op_, {out_grads.at(0), scalar}));
  }
  if (ctx->scalar_requires_grad) {
    const auto& scalar = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->at(1) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*broadcast_div_grad_op_, {out_grads.at(0), y, scalar}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_div_by_tensor", TensorScalarDiv);

}  // namespace one
}  // namespace oneflow
