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

#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ElementwiseXimumCaptureState : public AutoGradCaptureState {
  bool x_requires_grad;
  bool y_requires_grad;
};

class ElementwiseXimumOp : public OpExprGradFunction<ElementwiseXimumCaptureState> {
 public:
  Maybe<void> Capture(ElementwiseXimumCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    state->x_requires_grad = inputs.at(0)->requires_grad();
    state->y_requires_grad = inputs.at(1)->requires_grad();
    state->SaveTensorForBackward(inputs.at(0));
    state->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ElementwiseXimumCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!(state->x_requires_grad || state->y_requires_grad)) { return Maybe<void>::Ok(); }

    in_grads->resize(2);
    const std::shared_ptr<one::Tensor>& x = state->SavedTensors().at(0);
    const std::shared_ptr<one::Tensor>& y = state->SavedTensors().at(1);
    if (state->x_requires_grad || state->y_requires_grad) {
      const auto& grads = JUST(grad_functor(out_grads.at(0), x, y));
      if (state->x_requires_grad) { in_grads->at(0) = grads->at(0); }
      if (state->y_requires_grad) { in_grads->at(1) = grads->at(1); }
    }

    return Maybe<void>::Ok();
  }

 protected:
  std::function<Maybe<TensorTuple>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&,
                                   const std::shared_ptr<Tensor>&)>
      grad_functor;
};

class ElementwiseMinimum : public ElementwiseXimumOp {
 public:
  ElementwiseMinimum() : ElementwiseXimumOp(), grad_functor{functional::ElementwiseMinGrad} {}
};

class ElementwiseMaximum : public ElementwiseXimumOp {
 public:
  ElementwiseMaximum() : ElementwiseXimumOp(), grad_functor{functional::ElementwiseMaxGrad} {}
};

REGISTER_OP_EXPR_GRAD_FUNCTION("elementwise_minimum", ElementwiseMinimum);
REGISTER_OP_EXPR_GRAD_FUNCTION("elementwise_maximum", ElementwiseMaximum);

}  // namespace one
}  // namespace oneflow
