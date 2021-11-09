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
#include "oneflow/user/ops/math_binary_elementwise_seq.h"

namespace oneflow {
namespace one {

struct BinaryMathCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
  bool y_requires_grad;
};

typedef Maybe<one::Tensor> (*BinaryBwFunc)(const std::shared_ptr<one::Tensor>&,
                                           const std::shared_ptr<one::Tensor>&,
                                           const std::shared_ptr<one::Tensor>&);

template<BinaryBwFunc BwXFunc, BinaryBwFunc BwYFunc>
class BinaryMathOp : public OpExprGradFunction<BinaryMathCaptureState> {
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(BinaryMathCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const BinaryMathCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!(ctx->x_requires_grad || ctx->y_requires_grad)) { return Maybe<void>::Ok(); }
    in_grads->resize(2);
    const std::shared_ptr<one::Tensor>& x = ctx->SavedTensors().at(0);
    const std::shared_ptr<one::Tensor>& y = ctx->SavedTensors().at(1);
    if (ctx->x_requires_grad) { in_grads->at(0) = JUST(BwXFunc(x, y, out_grads.at(0))); }
    if (ctx->y_requires_grad) { in_grads->at(1) = JUST(BwYFunc(x, y, out_grads.at(0))); }
    return Maybe<void>::Ok();
  }
};

#define INSTANTIAT_AND_REGISTER_BINARY_MATHOP_CLASS(op_type_name, op_cls)             \
  class op_cls##Cls final                                                             \
      : public BinaryMathOp<functional::op_cls##XGrad, functional::op_cls##YGrad> {}; \
  REGISTER_OP_EXPR_GRAD_FUNCTION(op_type_name, op_cls##Cls);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_AND_REGISTER_BINARY_MATHOP_CLASS, MATH_BINARY_ELEMENTWISE_FUNC_SEQ);

#undef INSTANTIAT_AND_REGISTER_BINARY_MATHOP_CLASS
}  // namespace one
}  // namespace oneflow
