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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct ModeCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
};

class Mode : public OpExprGradFunction<ModeCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(ModeCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
    if (ctx->requires_grad) {
      ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));
      ctx->SaveTensorForBackward(JUST(VectorAt(outputs, 1)));
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const ModeCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (ctx->requires_grad) {
      in_grads->resize(1);
      const auto& input = JUST(VectorAt(ctx->SavedTensors(), 0));
      const auto& indices = JUST(functional::Unsqueeze(JUST(VectorAt(ctx->SavedTensors(), 1)), -1));
      const auto& dout = JUST(functional::Unsqueeze(JUST(VectorAt(out_grads, 0)), -1));
      JUST(VectorAt(*in_grads, 0)) = JUST(functional::DimScatterUpdate(
          JUST(functional::Constant(*(input->shape()), Scalar(0), *dout->dtype(),
                                    JUST(dout->device()))),
          -1, indices, dout, /*inplace*/ false));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("mode", Mode);

}  // namespace one
}  // namespace oneflow
