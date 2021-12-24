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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct MatmulCaptureState : public AutoGradCaptureState {
  bool transpose_a;
  bool transpose_b;
  double alpha;
  bool requires_grad_a;
  bool requires_grad_b;
  size_t a_index;
  size_t b_index;
};

class Matmul : public OpExprGradFunction<MatmulCaptureState> {
 public:
  Maybe<void> Capture(MatmulCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const MatmulCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Matmul::Capture(MatmulCaptureState* state, const TensorTuple& inputs,
                            const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad_a = inputs.at(0)->requires_grad();
  state->requires_grad_b = inputs.at(1)->requires_grad();
  if (!state->requires_grad_a && !state->requires_grad_b) { return Maybe<void>::Ok(); }

  auto* interp_ctx = dynamic_cast<const MatmulOp*>(ctx);
  state->transpose_a = interp_ctx->transpose_a();
  state->transpose_b = interp_ctx->transpose_b();
  state->alpha = interp_ctx->alpha();
  if (state->requires_grad_a) {
    state->b_index = state->SaveTensorForBackward(inputs.at(1));  // input b
  }
  if (state->requires_grad_b) {
    state->a_index = state->SaveTensorForBackward(inputs.at(0));  // input a
  }
  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Apply(const MatmulCaptureState* state, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!state->requires_grad_a && !state->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  in_grads->resize(2);
  if (state->requires_grad_a) {
    const auto& input_b = state->SavedTensors().at(state->b_index);
    if (state->transpose_a) {
      in_grads->at(0) = JUST(
          functional::MatMul(input_b, out_grads.at(0), state->transpose_b, true, state->alpha));
    } else {
      in_grads->at(0) = JUST(
          functional::MatMul(out_grads.at(0), input_b, false, !(state->transpose_b), state->alpha));
    }
  }

  if (state->requires_grad_b) {
    const auto& input_a = state->SavedTensors().at(state->a_index);
    if (state->transpose_b) {
      in_grads->at(1) = JUST(
          functional::MatMul(out_grads.at(0), input_a, true, state->transpose_a, state->alpha));
    } else {
      in_grads->at(1) = JUST(
          functional::MatMul(input_a, out_grads.at(0), !(state->transpose_a), false, state->alpha));
    }
  }

  return Maybe<void>::Ok();
}

class BroadcastMatmul : public Matmul {
 public:
  Maybe<void> Apply(const MatmulCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BroadcastMatmul::Apply(const MatmulCaptureState* state, const TensorTuple& out_grads,
                                   TensorTuple* in_grads) const {
  if (!state->requires_grad_a && !state->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  in_grads->resize(2);
  if (state->requires_grad_a) {
    const auto& input_b = state->SavedTensors().at(state->b_index);
    if (state->transpose_a) {
      in_grads->at(0) = JUST(
          functional::MatMul(input_b, out_grads.at(0), state->transpose_b, true, state->alpha));
    } else {
      in_grads->at(0) = JUST(
          functional::MatMul(out_grads.at(0), input_b, false, !(state->transpose_b), state->alpha));
    }
  }

  if (state->requires_grad_b) {
    const auto& input_a = state->SavedTensors().at(state->a_index);
    if (state->transpose_b) {
      in_grads->at(1) =
          JUST(functional::BroadcastMatmulGradB(out_grads.at(0), input_a, state->alpha));
    } else {
      in_grads->at(1) =
          JUST(functional::BroadcastMatmulGradB(input_a, out_grads.at(0), state->alpha));
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("batch_matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_matmul", BroadcastMatmul);

}  // namespace one
}  // namespace oneflow
