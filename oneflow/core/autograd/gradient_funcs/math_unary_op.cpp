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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct UnaryMathCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
};

typedef Maybe<one::Tensor> (*UnaryBwFunc)(const std::shared_ptr<one::Tensor>&,
                                          const std::shared_ptr<one::Tensor>&);

template<UnaryBwFunc BwFunc>
class UnaryMathBwdWithDyXOp : public OpExprGradFunction<UnaryMathCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryMathCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryMathCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(BwFunc(x, out_grads.at(0)));
    return Maybe<void>::Ok();
  }

 protected:
  std::shared_ptr<OpExpr> grad_op_;
};

template<UnaryBwFunc BwFunc>
class UnaryMathBwdWithDyYOp : public OpExprGradFunction<UnaryMathCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryMathCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryMathCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    const auto& y = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(BwFunc(y, out_grads.at(0)));
    return Maybe<void>::Ok();
  }

 protected:
  std::shared_ptr<OpExpr> grad_op_;
};

class UnaryMathBwdWithFillZeroOp : public OpExprGradFunction<UnaryMathCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryMathCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryMathCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    in_grads->at(0) = JUST(functional::ZerosLike(out_grads[0]));
    return Maybe<void>::Ok();
  }

 protected:
  std::shared_ptr<OpExpr> grad_op_;
};

#define INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_X_CLASS(op_type_name, op_cls)     \
  class op_cls##Cls final : public UnaryMathBwdWithDyXOp<functional::op_cls##Grad> {}; \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, op_cls##Cls);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_X_CLASS,
                     MATH_UNARY_ELEMENTWISE_PRIMITIVE_FUNC_BWD_WITH_DY_X_SEQ);
OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_X_CLASS,
                     OF_PP_MAKE_TUPLE_SEQ("tanh", Tanh));

#undef INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_X_CLASS

#define INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_Y_CLASS(op_type_name, op_cls)     \
  class op_cls##Cls final : public UnaryMathBwdWithDyYOp<functional::op_cls##Grad> {}; \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, op_cls##Cls);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_Y_CLASS,
                     MATH_UNARY_ELEMENTWISE_FUNC_BWD_WITH_DY_Y_SEQ);
#undef INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_DY_Y_CLASS

#define INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_FILL_CLASS(op_type_name, op_cls)     \
  class op_cls##Cls final : public UnaryMathBwdWithDyYOp<functional::op_cls##Grad> {}; \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, UnaryMathBwdWithFillZeroOp);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_FILL_CLASS,
                     MATH_UNARY_ELEMENTWISE_FUNC_BWD_WITH_FILL_SEQ);
#undef INSTANTIAT_AND_REGISTER_UNARY_MATHOP_WITH_FILL_CLASS

class NegativeOp : public OpExprGradFunction<UnaryMathCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UnaryMathCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UnaryMathCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    in_grads->at(0) = JUST(functional::Negative(out_grads[0]));
    return Maybe<void>::Ok();
  }

 protected:
  std::shared_ptr<OpExpr> grad_op_;
};
REGISTER_OP_EXPR_GRAD_FUNCTION("negative", NegativeOp);

}  // namespace one
}  // namespace oneflow
