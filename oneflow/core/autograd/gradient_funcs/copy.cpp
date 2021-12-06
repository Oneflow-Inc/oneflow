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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct CopyCaptureState : public AutoGradCaptureState {
  std::string device_type;
  int64_t device_id;
};

class Copy : public OpExprGradFunction<CopyCaptureState> {
 public:
  Maybe<void> Capture(CopyCaptureState* state, const TensorTuple& inputs, const TensorTuple& outputs,
                      const OpInterpCtx* ctx) const override {
    state->device_type = JUST(inputs.at(0)->device())->type();
    state->device_id = JUST(inputs.at(0)->device())->device_id();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CopyCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::Copy(out_grads.at(0), state->device_type, state->device_id));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("copy", Copy);

}  // namespace one
}  // namespace oneflow
