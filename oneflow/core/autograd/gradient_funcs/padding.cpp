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

struct Pad2dCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::vector<int64_t> paddings;
};

class Pad2d : public OpExprGradFunction<Pad2dCaptureState> {
 public:
  Maybe<void> Capture(Pad2dCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = dynamic_cast<const ReflectionPad2DOp*>(ctx);
    state->paddings = op_ctx->padding();
    return Maybe<void>::Ok();
  }
};

class ReflectionPad2d : public Pad2d {
 public:
  Maybe<void> Apply(const Pad2dCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      in_grads->at(0) = JUST(functional::PadGrad(out_grads.at(0), state->paddings, "reflect", 0));
    }
    return Maybe<void>::Ok();
  }
};

class ReplicationPad2d : public Pad2d {
 public:
  Maybe<void> Apply(const Pad2dCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      in_grads->at(0) = JUST(functional::PadGrad(out_grads.at(0), state->paddings, "replicate", 0));
    }
    return Maybe<void>::Ok();
  }
};

struct ConstantPadNdCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::vector<int64_t> paddings;
  Scalar padding_value;
};

class ConstantPadNd : public OpExprGradFunction<ConstantPadNdCaptureState> {
 public:
  Maybe<void> Capture(ConstantPadNdCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }

    auto* op_ctx = dynamic_cast<const PadOp*>(ctx);
    const auto& pad_before = op_ctx->padding_before();
    const auto& pad_after = op_ctx->padding_after();

    if (pad_before.size() != pad_after.size()) {
      return Error::RuntimeError() << "padding_before and padding_after size mismatch";
    }
    int64_t size = pad_before.size();
    state->paddings.resize(size * 2);
    for (int64_t i = 0; i < size; ++i) {
      state->paddings[2 * i] = pad_before[size - i - 1];
      state->paddings[2 * i + 1] = pad_after[size - i - 1];
    }
    if (IsFloatingDataType(inputs.at(0)->dtype()->data_type())) {
      state->padding_value = op_ctx->floating_constant_value();
    } else if (IsIntegralDataType(inputs.at(0)->dtype()->data_type())) {
      state->padding_value = op_ctx->integral_constant_value();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Data type should be floating or integral type.";
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ConstantPadNdCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (state->requires_grad) {
      in_grads->at(0) = JUST(
          functional::PadGrad(out_grads.at(0), state->paddings, "constant", state->padding_value));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("pad", ConstantPadNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("reflection_pad2d", ReflectionPad2d);
REGISTER_OP_EXPR_GRAD_FUNCTION("replication_pad2d", ReplicationPad2d);

}  // namespace one
}  // namespace oneflow
