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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct ReduceSumCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
};

class ReduceSum : public OpExprGradFunction<ReduceSumCaptureState> {
 public:
  Maybe<void> Capture(ReduceSumCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const ReduceSumCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> ReduceSum::Capture(ReduceSumCaptureState* state, const TensorTuple& inputs,
                               const TensorTuple& outputs, const OpBase* ctx) const {
  auto* op_ctx = dynamic_cast<const ReduceSumOp*>(ctx);
  state->axis = op_ctx->axis();
  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSum::Apply(const ReduceSumCaptureState* state, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  const auto& input = state->SavedTensors().at(0);
  const auto& dy = out_grads.at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::BroadcastLike(dy, input, state->axis));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_sum", ReduceSum);

struct ReduceProdInterpState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad;
};

class ReduceProd : public OpExprGradFunction<ReduceProdInterpState> {
 public:
  Maybe<void> Capture(ReduceProdInterpState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const ReduceProdInterpState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> ReduceProd::Capture(ReduceProdInterpState* state, const TensorTuple& inputs,
                                const TensorTuple& outputs, const OpBase* ctx) const {
  auto* op_ctx = dynamic_cast<const ReduceProdOp*>(ctx);
  state->axis = op_ctx->axis();
  state->requires_grad = inputs.at(0)->requires_grad();
  state->SaveTensorForBackward(inputs.at(0));
  state->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceProd::Apply(const ReduceProdInterpState* state, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  const auto& input = state->SavedTensors().at(0);
  const auto& output = state->SavedTensors().at(1);
  const auto& dy = out_grads.at(0);

  in_grads->resize(1);
  in_grads->at(0) = JUST(
      functional::SequenceFunction<Maybe<Tensor>()>([&]() { return functional::Mul(dy, output); })
          .then(std::bind(functional::BroadcastLike, std::placeholders::_1, input, state->axis))
          .then(std::bind(functional::Div, std::placeholders::_1, input))
          .call());
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_prod", ReduceProd);

struct ReduceMaxOrMinCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool keepdims;
};

class ReduceMaxOrMin : public OpExprGradFunction<ReduceMaxOrMinCaptureState> {
 public:
  Maybe<void> Capture(ReduceMaxOrMinCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const ReduceMaxOrMinCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> ReduceMaxOrMin::Capture(ReduceMaxOrMinCaptureState* state, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const OpBase* ctx) const {
  auto* op_ctx = dynamic_cast<const ReduceMaxOp*>(ctx);
  state->axis = op_ctx->axis();
  state->keepdims = op_ctx->keepdims();
  state->SaveTensorForBackward(inputs.at(0));
  state->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> ReduceMaxOrMin::Apply(const ReduceMaxOrMinCaptureState* state,
                                  const TensorTuple& out_grads, TensorTuple* in_grads) const {
  const auto& input = state->SavedTensors().at(0);
  const auto& output = state->SavedTensors().at(1);
  const auto& dy = out_grads.at(0);

  const auto cast_like =
      JUST(functional::SequenceFunction<Maybe<Tensor>()>(
               [&]() { return functional::BroadcastLike(output, input, state->axis); })
               .then(std::bind(functional::BroadcastEqual, input, std::placeholders::_1))
               .then(std::bind(functional::CastLike, std::placeholders::_1, input))
               .call());

  const auto& bcast_like_div = JUST(
      functional::SequenceFunction<Maybe<Tensor>()>(
          [&]() { return functional::ReduceSum(cast_like, state->axis, state->keepdims); })
          .then(std::bind(functional::Div, dy, std::placeholders::_1))
          .then(std::bind(functional::BroadcastLike, std::placeholders::_1, input, state->axis))
          .call());

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Mul(bcast_like_div, cast_like));

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min", ReduceMaxOrMin);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max", ReduceMaxOrMin);

}  // namespace one
}  // namespace oneflow
