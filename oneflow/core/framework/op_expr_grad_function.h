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

#include "oneflow/core/autograd/autograd_captured_tensor.h"
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
    saved_tensors_.emplace_back(tensor);
    return offset;
  }

 protected:
  TensorTuple saved_tensors_;
};

class FunctionAutoGradCaptureState final
    : public AutoGradCaptureState,
      public std::enable_shared_from_this<FunctionAutoGradCaptureState> {
 public:
  FunctionAutoGradCaptureState() = default;
  using AutoGradCaptureState::SavedTensors;
  using AutoGradCaptureState::SaveTensorForBackward;

  void MarkNonDifferentiable(const std::shared_ptr<Tensor>& tensor) {
    non_differentiable_tensors_.emplace(tensor.get());
  }

  HashSet<Tensor*> NonDifferentiableTensors() const { return non_differentiable_tensors_; }

  std::shared_ptr<FunctionAutoGradCaptureState> GetSharedFromThis() { return shared_from_this(); }

 private:
  HashSet<Tensor*> non_differentiable_tensors_;
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
    // Convert outputs from `Tensor` to `AutogradCapturedTensor` to avoid
    // circular reference between `Tensor` and `FunctionNode`.
    TensorTuple captured_inputs(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      captured_inputs[i] = JUST(AutogradCapturedTensor::MakeTensor(inputs.at(i)));
    }
    TensorTuple captured_outputs(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      captured_outputs[i] = JUST(AutogradCapturedTensor::MakeTensor(outputs.at(i)));
    }
    return Capture(state, captured_inputs, captured_outputs, interp_ctx);
  }

  Maybe<void> ApplyIf(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                      TensorTuple* in_grads) const override {
    const StateT* state = dynamic_cast<const StateT*>(ctx);
    CHECK_NOTNULL_OR_RETURN(state);
    return Apply(state, out_grads, in_grads);
  }

 protected:
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

  std::string GradientOpName(const std::string& prefix) const {
    return prefix + std::string(kGradientOpSuffix);
  }
};

class FunctionOpExprGradFunction final : public OpExprGradFunctionIf {
 public:
  using FType = AutogradFunctionBase::FType;
  explicit FunctionOpExprGradFunction(const FType& backward_fn) : backward_fn_(backward_fn) {}

  std::shared_ptr<AutoGradCaptureState> MakeCustomState() const override {
    PRINT_BUG_PROMPT_AND_ABORT()
        << "You should not construct AutoGradCaptureState by calling this function";
    return std::make_shared<FunctionAutoGradCaptureState>();
  }

  Maybe<void> Init(const OpExpr& op) override {
    // do nothing
    return Maybe<void>::Ok();
  }

  Maybe<void> CaptureIf(AutoGradCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs,
                        const OpExprInterpContext& interp_ctx) const override {
    // do nothing
    return Maybe<void>::Ok();
  }

  Maybe<void> ApplyIf(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                      TensorTuple* in_grads) const override {
    const FunctionAutoGradCaptureState* func_ctx =
        dynamic_cast<const FunctionAutoGradCaptureState*>(ctx);
    CHECK_NOTNULL_OR_RETURN(func_ctx);
    const std::shared_ptr<TensorTuple>& out = backward_fn_(
        const_cast<FunctionAutoGradCaptureState*>(func_ctx)->GetSharedFromThis(), out_grads);
    in_grads->assign(out->begin(), out->end());
    return Maybe<void>::Ok();
  }

 protected:
  FType backward_fn_;
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

  const std::shared_ptr<AutoGradCaptureState>& state() const { return state_; }

 private:
  std::shared_ptr<OpExprGradFunctionIf> impl_;
  std::shared_ptr<AutoGradCaptureState> state_;
};

#define REGISTER_OP_EXPR_GRAD_FUNCTION(op_type, op_grad) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpExprGradFunctionIf, ([]() { return new op_grad; }))

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_FUNCTION_H_
