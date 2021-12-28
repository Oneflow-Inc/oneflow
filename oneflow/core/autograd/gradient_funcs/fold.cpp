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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FoldInterpState : public AutoGradCaptureState {
  bool requires_grad = true;
  std::string data_format = "channels_first";
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> dilation_rate;
  std::vector<int32_t> padding;
  std::vector<int32_t> strides;
};

class Fold : public OpExprGradFunction<FoldInterpState> {
 public:
  Maybe<void> Capture(FoldInterpState* state, const TensorTuple& inputs, const TensorTuple& outputs,
                      const OpBase* ctx) const override;
  Maybe<void> Apply(const FoldInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Fold::Capture(FoldInterpState* state, const TensorTuple& inputs,
                          const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  auto* op_ctx = JUST(ctx->dyn_cast<FoldOp>());
  state->kernel_size = op_ctx->kernel_size();
  state->dilation_rate = op_ctx->dilation_rate();
  state->padding = op_ctx->padding();
  state->strides = op_ctx->strides();
  return Maybe<void>::Ok();
}

Maybe<void> Fold::Apply(const FoldInterpState* state, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Unfold(out_grads.at(0), state->data_format, state->kernel_size,
                                            state->dilation_rate, state->padding, state->strides));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fold", Fold);

}  // namespace one
}  // namespace oneflow
