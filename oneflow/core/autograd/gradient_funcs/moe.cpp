/*
Copyright 2022 The OneFlow Authors. All rights reserved.

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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct MOECaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool gate_requires_grad = false;
  bool has_gate = false;
  int num_experts = 0;
  int capacity = 0;
};

class MOEDispatch : public OpExprGradFunction<MOECaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MOECaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MOECaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

class MOECombine : public OpExprGradFunction<MOECaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MOECaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MOECaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

/********** MOECDispatch Gradient ***********/
Maybe<void> MOEDispatch::Init(const OpExpr& op) {return Maybe<void>::Ok(); }

Maybe<void> MOEDispatch::Capture(MOECaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() >= 3 && inputs.size() <= 4);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = inputs[0]->requires_grad();
  ctx->has_gate = inputs.size() == 4;
  if (ctx->has_gate) {
    ctx->gate_requires_grad = inputs[3]->requires_grad();
  }

  // without gate
  // data_grad: indices, locations, out_grad
  // with gate
  // data_grad: gates, indices, locations, out_grad
  // gate_grad: indices, locations, input, out_grad
  ctx->SaveTensorForBackward(inputs[1]);  // indices
  ctx->SaveTensorForBackward(inputs[2]);  // locations
  if (ctx->has_gate) {
    ctx->SaveTensorForBackward(inputs[0]);  // in
    ctx->SaveTensorForBackward(inputs[3]);  // gate
  }
  return Maybe<void>::Ok();
}

Maybe<void> MOEDispatch::Apply(const MOECaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(),
                     ctx->has_gate ? 4 : 2);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3 + ctx->has_gate);

  const auto& dout = out_grads[0];
  const auto& indices = ctx->SavedTensors()[0];
  const auto& locations = ctx->SavedTensors()[1];
  const auto& gates = ctx->has_gate ? Optional<one::Tensor>(ctx->SavedTensors()[3]) : NullOpt;

  (*in_grads)[0] = JUST(functional::MOECombine(dout, indices, locations, gates));
  if (ctx->gate_requires_grad) {
    const auto& in = ctx->SavedTensors()[2];
    (*in_grads)[3] = JUST(functional::MOEGateGrad(in, dout, indices, locations));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("moe_dispatch", MOEDispatch);

/********** MOECombine Gradient ***********/
Maybe<void> MOECombine::Init(const OpExpr& op) {return Maybe<void>::Ok(); }

Maybe<void> MOECombine::Capture(MOECaptureState* ctx, const TensorTuple& inputs,
                                const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() >= 3 && inputs.size() <= 4);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = inputs[0]->requires_grad();
  ctx->has_gate = inputs.size() == 4;
  if (ctx->has_gate) {
    ctx->gate_requires_grad = inputs[3]->requires_grad();
  }

  // without gate
  // data_grad: indices, locations, out_grad
  // with gate
  // data_grad: gates, indices, locations, out_grad
  // gate_grad: indices, locations, out_grad, input
  ctx->SaveTensorForBackward(inputs[1]);  // indices
  ctx->SaveTensorForBackward(inputs[2]);  // locations
  if (ctx->has_gate) {
    ctx->SaveTensorForBackward(inputs[0]);  // in
    ctx->SaveTensorForBackward(inputs[3]);  // gate
  }
  ctx->num_experts = inputs[0]->shape()->At(0);
  ctx->capacity = inputs[0]->shape()->At(1);
  return Maybe<void>::Ok();
}

Maybe<void> MOECombine::Apply(const MOECaptureState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(),
                     ctx->has_gate ? 4 : 2);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(3 + ctx->has_gate);

  const auto& dout = out_grads[0];
  const auto& indices = ctx->SavedTensors()[0];
  const auto& locations = ctx->SavedTensors()[1];
  const auto& gates = ctx->has_gate ? Optional<one::Tensor>(ctx->SavedTensors()[3]) : NullOpt;

  (*in_grads)[0] = JUST(
      functional::MOEDispatch(dout,indices, locations, gates, ctx->num_experts, ctx->capacity));
  if (ctx->gate_requires_grad) {
    const auto& in = ctx->SavedTensors()[2];
    (*in_grads)[3] = JUST(functional::MOEGateGrad(dout, in, indices, locations));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("moe_combine", MOECombine);

}  // namespace one
}  // namespace oneflow
