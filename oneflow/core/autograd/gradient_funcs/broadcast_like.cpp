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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BroadCastLikeCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  size_t input_index;

  std::vector<int32_t> broadcast_axes;
};

class BroadCastLike : public OpExprGradFunction<BroadCastLikeCaptureState> {
 public:
  Maybe<void> Capture(BroadCastLikeCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const BroadCastLikeCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> BroadCastLike::Capture(BroadCastLikeCaptureState* state, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  auto* op_ctx = dynamic_cast<const BroadcastLikeOp*>(ctx);
  state->broadcast_axes = op_ctx->broadcast_axes();
  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> BroadCastLike::Apply(const BroadCastLikeCaptureState* state,
                                 const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const auto& x = state->SavedTensors().at(state->input_index);
  in_grads->resize(2);
  in_grads->at(0) = JUST(functional::ReduceSumLike(out_grads.at(0), x, state->broadcast_axes));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_like", BroadCastLike);

}  // namespace one
}  // namespace oneflow
