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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/system_ops.h"

namespace oneflow {
namespace one {

struct SelectTopNCaptureState : public AutoGradCaptureState {
  TensorTuple inputs;
  std::vector<bool> requires_grad;
  int32_t top_n = 0;
};

class SelectTopN : public OpExprGradFunction<SelectTopNCaptureState> {
 public:
  Maybe<void> Capture(SelectTopNCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    auto* op_ctx = dynamic_cast<const schema::SelectTopNOp*>(ctx);
    state->inputs = inputs;
    state->top_n = op_ctx->top_n;
    state->requires_grad.resize(inputs.size());
    for (int i = 0; i < state->requires_grad.size(); ++i) {
      state->requires_grad.at(i) = inputs.at(i)->requires_grad();
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SelectTopNCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(state->top_n, out_grads.size());
    for (int i = 0; i < state->top_n; ++i) {
      if (!state->requires_grad.at(i)) { continue; }
      in_grads->at(i) = out_grads.at(i);
    }
    for (int i = state->top_n; i < state->inputs.size(); ++i) {
      if (!state->requires_grad.at(i)) { continue; }
      const auto& tensor = state->inputs.at(i);
      in_grads->at(i) = JUST(StaticZerosTensor::MakeTensor(
          tensor->shape(), tensor->dtype()->data_type(), JUST(tensor->device())));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("select_top_n", SelectTopN);

}  // namespace one
}  // namespace oneflow
