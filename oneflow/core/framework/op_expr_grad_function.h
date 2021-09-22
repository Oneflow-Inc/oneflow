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

#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_FUNCTION_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_FUNCTION_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {
namespace one {

static constexpr char kGradientOpSuffix[] = ".grad";

class AutoGradCaptureState {
 public:
  AutoGradCaptureState() = default;
  virtual ~AutoGradCaptureState() = default;

  const TensorTuple& SavedTensors() const { return saved_tensors_; }

  size_t SaveTensorForBackward(const std::shared_ptr<Tensor>& tensor) {
    size_t offset = saved_tensors_.size();
    saved_tensors_.push_back(tensor);
    return offset;
  }

 private:
  TensorTuple saved_tensors_;
};

// Stateless container base of the backward op exprs.
// The backward op exprs should be contained in the derived class.
class OpExprGradFunctionIf {
 public:
  virtual ~OpExprGradFunctionIf() = default;

  virtual std::shared_ptr<AutoGradCaptureState> MakeCustomState() const = 0;

  virtual Maybe<void> Init(const OpExpr& op) = 0;

  // Capture forward inputs and outputs for backward.
  virtual Maybe<void> CaptureIf(AutoGradCaptureState* ctx, const TensorTuple& inputs,
                                const TensorTuple& outputs,
                                const OpExprInterpContext& interp_ctx) const = 0;

  virtual Maybe<void> ApplyIf(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const = 0;
};

template<typename StateT>
class OpExprGradFunction : public OpExprGradFunctionIf {
 public:
  std::shared_ptr<AutoGradCaptureState> MakeCustomState() const override {
    return std::make_shared<StateT>();
  }

  Maybe<void> CaptureIf(AutoGradCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs,
                        const OpExprInterpContext& interp_ctx) const override {
    StateT* state = dynamic_cast<StateT*>(ctx);
    CHECK_NOTNULL_OR_RETURN(state);
    // Only captures detached tensor for calculating grad and set right requires_grad
    // for higher order derivative
    TensorTuple detach_inputs(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      detach_inputs.at(i) = JUST(inputs.at(i)->detach());
      JUST(detach_inputs.at(i)->set_requires_grad(inputs.at(i)->requires_grad()));
    }
    TensorTuple detach_outputs(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      detach_outputs.at(i) = JUST(outputs.at(i)->detach());
      JUST(detach_outputs.at(i)->set_requires_grad(outputs.at(i)->requires_grad()));
    }
    return Capture(state, detach_inputs, detach_outputs, interp_ctx);
  }

  Maybe<void> ApplyIf(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                      TensorTuple* in_grads) const override {
    const StateT* state = dynamic_cast<const StateT*>(ctx);
    CHECK_NOTNULL_OR_RETURN(state);
    return Apply(state, out_grads, in_grads);
  }

  virtual Maybe<void> Capture(StateT* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                              const OpExprInterpContext& interp_ctx) const {
    return Capture(ctx, inputs, outputs, interp_ctx.attrs);
  }

  virtual Maybe<void> Capture(StateT* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                              const AttrMap& attrs) const {
    UNIMPLEMENTED_THEN_RETURN();
  }

  virtual Maybe<void> Apply(const StateT* ctx, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const = 0;

 protected:
  std::string GradientOpName(const std::string& prefix) const {
    return prefix + std::string(kGradientOpSuffix);
  }
};

// Stateful wrapper of the `OpExprGradFunction`.
class OpExprGradClosure {
 public:
  // Use `shared_ptr` in order to keep `impl` alive even if the forward op has been released.
  explicit OpExprGradClosure(const std::shared_ptr<OpExprGradFunctionIf>& impl)
      : impl_(impl), state_(impl->MakeCustomState()) {}
  explicit OpExprGradClosure(const std::shared_ptr<OpExprGradFunctionIf>& impl,
                             const std::shared_ptr<AutoGradCaptureState>& state)
      : impl_(impl), state_(state) {}

  virtual ~OpExprGradClosure() = default;

  Maybe<void> Capture(const TensorTuple& inputs, const TensorTuple& outputs,
                      const OpExprInterpContext& interp_ctx) const {
    return impl_->CaptureIf(state_.get(), inputs, outputs, interp_ctx);
  }

  Maybe<void> Apply(const TensorTuple& out_grads, TensorTuple* in_grads) const {
    return impl_->ApplyIf(state_.get(), out_grads, in_grads);
  }

 private:
  std::shared_ptr<OpExprGradFunctionIf> impl_;
  std::shared_ptr<AutoGradCaptureState> state_;
};

#define REGISTER_OP_EXPR_GRAD_FUNCTION(op_type, op_grad) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpExprGradFunctionIf, ([]() { return new op_grad; }))

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_FUNCTION_H_
