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

#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BiasAddCaptureState : public AutoGradCaptureState {
  bool input_requires_grad;
  bool bias_requires_grad;
  int32_t axis;
};

class BiasAdd : public OpExprGradFunction<BiasAddCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(BiasAddCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    state->input_requires_grad = inputs.at(0)->requires_grad();
    state->bias_requires_grad = inputs.at(1)->requires_grad();
    auto* interp_ctx = dynamic_cast<const BiasAddOpInterpCtx*>(ctx);
    state->axis = interp_ctx->axis;
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const BiasAddCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const int64_t num_axes = out_grads.at(0)->shape()->NumAxes();
    in_grads->resize(2);
    if (state->bias_requires_grad) {
      std::vector<int32_t> reduce_axes_vec;
      reduce_axes_vec.reserve(num_axes);
      for (int i = 0; i < num_axes; ++i) {
        if (i != state->axis) { reduce_axes_vec.push_back(i); }
      }
      if (state->bias_requires_grad) {
        in_grads->at(1) = JUST(functional::ReduceSum(out_grads.at(0), reduce_axes_vec, false));
      }
    }
    if (state->input_requires_grad) { in_grads->at(0) = out_grads.at(0); }

    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("bias_add", BiasAdd);

}  // namespace one
}  // namespace oneflow
