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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct UnaryMathGradGradState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;
};

typedef Maybe<one::Tensor> (*UnaryBwFunc)(const std::shared_ptr<one::Tensor>&,
                                          const std::shared_ptr<one::Tensor>&);

template<UnaryBwFunc BwFunc, UnaryBwFunc BwBwFunc>
class UnaryMathGradGrad : public OpExprGradFunction<UnaryMathGradGradState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(UnaryMathGradGradState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs[0]->requires_grad();
    ctx->grad_requires_grad = inputs[1]->requires_grad();
    ctx->SaveTensorForBackward(inputs[0]);
    if (ctx->x_requires_grad) { ctx->SaveTensorForBackward(inputs[1]); }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const UnaryMathGradGradState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors()[0];
    if (ctx->x_requires_grad) {
      const auto& grad = ctx->SavedTensors()[1];
      (*in_grads)[0] = JUST(functional::Mul(out_grads[0], JUST(BwBwFunc(x, grad))));
    }
    if (ctx->grad_requires_grad) { (*in_grads)[1] = JUST(BwFunc(x, out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

template<UnaryBwFunc BwFunc>
class UnaryMathGradGradWithZeroDDX : public OpExprGradFunction<UnaryMathGradGradState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(UnaryMathGradGradState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs[0]->requires_grad();
    ctx->grad_requires_grad = inputs[1]->requires_grad();
    ctx->SaveTensorForBackward(inputs[0]);
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const UnaryMathGradGradState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    const auto& x = ctx->SavedTensors()[0];
    if (ctx->x_requires_grad) { (*in_grads)[0] = JUST(functional::ZerosLike(x)); }
    if (ctx->grad_requires_grad) { (*in_grads)[1] = JUST(BwFunc(x, out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

#define MATH_UNARY_ELEMENTWISE_GRAD_GRAD_DY_X_FUNC_SEQ \
  OF_PP_MAKE_TUPLE_SEQ("sin_grad", Sin)                \
  OF_PP_MAKE_TUPLE_SEQ("cos_grad", Cos)                \
  OF_PP_MAKE_TUPLE_SEQ("tan_grad", Tan)                \
  OF_PP_MAKE_TUPLE_SEQ("sinh_grad", Sinh)              \
  OF_PP_MAKE_TUPLE_SEQ("cosh_grad", Cosh)              \
  OF_PP_MAKE_TUPLE_SEQ("tanh_grad", Tanh)              \
  OF_PP_MAKE_TUPLE_SEQ("asin_grad", Asin)              \
  OF_PP_MAKE_TUPLE_SEQ("acos_grad", Acos)              \
  OF_PP_MAKE_TUPLE_SEQ("atan_grad", Atan)              \
  OF_PP_MAKE_TUPLE_SEQ("asinh_grad", Asinh)            \
  OF_PP_MAKE_TUPLE_SEQ("acosh_grad", Acosh)            \
  OF_PP_MAKE_TUPLE_SEQ("atanh_grad", Atanh)

#define INSTANTIAT_AND_REGISTER_UNARY_MATHOP_GRAD_GRAD_CLASS(op_type_name, op_cls)           \
  class op_cls##GradGradCls final                                                            \
      : public UnaryMathGradGrad<functional::op_cls##Grad, functional::op_cls##GradGrad> {}; \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, op_cls##GradGradCls);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_UNARY_MATHOP_GRAD_GRAD_CLASS,
                     MATH_UNARY_ELEMENTWISE_GRAD_GRAD_DY_X_FUNC_SEQ);

}  // namespace one
}  // namespace oneflow
