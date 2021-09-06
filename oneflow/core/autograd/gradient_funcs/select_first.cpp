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
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

struct SelectFirstCaptureState : public AutoGradCaptureState {
  TensorTuple inputs;
  std::vector<bool> requires_grad;
};

class SelectFirst : public OpExprGradFunction<SelectFirstCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(SelectFirstCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->inputs = inputs;
    CHECK_OR_RETURN(ctx->inputs.at(0)->requires_grad());
    ctx->requires_grad.resize(inputs.size());
    for (int i = 0; i < ctx->requires_grad.size(); ++i) {
      ctx->requires_grad.at(i) = inputs.at(i)->requires_grad();
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SelectFirstCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->at(0) = out_grads.at(0);
    for (int i = 1; i < ctx->inputs.size(); i++) {
      if (!ctx->requires_grad.at(i)) { continue; }
      const auto& tensor = ctx->inputs.at(i);
      in_grads->at(i) = JUST(StaticZerosTensor::MakeTensor(
          tensor->shape(), tensor->dtype()->data_type(), JUST(tensor->device())));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("select_first", SelectFirst);

}  // namespace one
}  // namespace oneflow
