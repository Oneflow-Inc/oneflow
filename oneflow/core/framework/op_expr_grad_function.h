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
#include "oneflow/core/framework/op_interpreter.h"  // OpExprInterpState

namespace oneflow {
namespace one {

static constexpr char kGradientOpSuffix[] = ".grad";

// Stateless container base of the backward op exprs.
// The backward op exprs should be contained in the derived class.
class OpExprGradFunction {
 public:
  OpExprGradFunction() = default;
  virtual ~OpExprGradFunction() = default;

  virtual Maybe<void> Init(const OpExpr& op) = 0;

  // Capture forward inputs and outputs for backward.
  virtual Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                              const TensorTuple& outputs) const = 0;

  virtual Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
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
  explicit OpExprGradClosure(const std::shared_ptr<OpExprGradFunction>& impl)
      : impl_(impl), state_(new OpExprInterpState) {}
  explicit OpExprGradClosure(const std::shared_ptr<OpExprGradFunction>& impl,
                             const std::shared_ptr<OpExprInterpState>& state)
      : impl_(impl), state_(state) {}

  virtual ~OpExprGradClosure() = default;

  Maybe<void> Capture(const TensorTuple& inputs, const TensorTuple& outputs) const {
    return impl_->Capture(state_.get(), inputs, outputs);
  }

  Maybe<void> Apply(const TensorTuple& out_grads, TensorTuple* in_grads) const {
    return impl_->Apply(state_.get(), out_grads, in_grads);
  }

 private:
  std::shared_ptr<OpExprGradFunction> impl_;
  std::shared_ptr<OpExprInterpState> state_;
};

#define REGISTER_OP_EXPR_GRAD_FUNCTION(op_type, op_grad) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpExprGradFunction, ([]() { return new op_grad; }))

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_FUNCTION_H_
