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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_H_

#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {
namespace one {

class OpExprInterpState {
 public:
  DEFINE_DEFAULT_CONSTRUCTOR(OpExprInterpState);

  const TensorList& SavedTensors() const { return saved_tensors_; }

  void SaveTensorForBackward(const TensorRef& tensor) { saved_tensors_.push_back(tensor); }

 private:
  TensorList saved_tensors_;
};

typedef struct OpExprInterpContext {
  Scope* scope;
  bool is_mirrored_strategy_enabled;
} OpExprInterpContext, *OpExprInterpContextPtr;

class OpExprInterpreter {
 public:
  OpExprInterpreter() : self_state_(new OpExprInterpState) {}
  virtual ~OpExprInterpreter() = default;

  virtual void Apply(const OpExpr* op, const TensorList& inputs, TensorList& outputs,
                     const OpExprInterpState* state) = 0;

  void ResetSelfState();
  std::shared_ptr<OpExprInterpState> state() const { return self_state_; }

 protected:
  std::shared_ptr<OpExprInterpState> self_state_;
};

class NormalInterpreter : public OpExprInterpreter {
 public:
  NormalInterpreter() = delete;
  NormalInterpreter(const OpExprInterpContext* context) : OpExprInterpreter(), context_(context) {}
  virtual ~NormalInterpreter() = default;

  void Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprInterpState* state) override;

  const OpExprInterpContext* context() const { return context_; }

#define DEFINE_NORMAL_VIRTUAL_APPLY_FUNC(op_type)                                            \
  virtual void Apply_(const op_type* op_expr, const TensorList& inputs, TensorList& outputs, \
                      const OpExprInterpState* state) = 0;

  DEFINE_NORMAL_VIRTUAL_APPLY_FUNC(UserOpExpr);
  DEFINE_NORMAL_VIRTUAL_APPLY_FUNC(FunctionOpExpr);
#undef DEFINE_NORMAL_VIRTUAL_APPLY_FUNC

 protected:
  const OpExprInterpContext* context_ = nullptr;
};

#define DECLARE_NORMAL_APPLY_FUNC(op_type)                                                   \
  virtual void Apply_(const op_type* op_expr, const TensorList& inputs, TensorList& outputs, \
                      const OpExprInterpState* state);

class LazyInterpreter : public NormalInterpreter {
 public:
  LazyInterpreter(const OpExprInterpContext* context) : NormalInterpreter(context) {}

 private:
  DECLARE_NORMAL_APPLY_FUNC(UserOpExpr);
  DECLARE_NORMAL_APPLY_FUNC(FunctionOpExpr);
};

class EagerInterpreter : public NormalInterpreter {
 public:
  EagerInterpreter(const OpExprInterpContext* context) : NormalInterpreter(context) {}

 private:
  DECLARE_NORMAL_APPLY_FUNC(UserOpExpr);
  DECLARE_NORMAL_APPLY_FUNC(FunctionOpExpr);
};

#undef DECLARE_NORMAL_APPLY_FUNC

class AutogradInterpreter : public OpExprInterpreter {
 public:
  AutogradInterpreter() = delete;
  AutogradInterpreter(NormalInterpreter* normal_interp)
      : OpExprInterpreter(), normal_interp_(normal_interp) {}

  void Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprInterpState* state) override;

 private:
  NormalInterpreter* normal_interp_ = nullptr;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_H_
