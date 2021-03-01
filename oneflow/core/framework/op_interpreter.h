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

using TensorList = std::vector<std::shared_ptr<Tensor>>;

class OpExprInterpState {
 public:
  const TensorList& SavedTensors() const { return saved_tensors_; }

  void SaveTensorForBackward(const std::shared_ptr<Tensor>& tensor) {
    saved_tensors_.push_back(tensor);
  }

 private:
  TensorList saved_tensors_;
};

typedef struct OpExprInterpContext {
  // const Scope* scope;
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

 private:
  std::shared_ptr<OpExprInterpState> self_state_;
};

#define FOR_ALL_OPS(_macro)   \
  _macro(UserOp);             \
  _macro(VariableOp);         \
  _macro(CastToMirroredOp);   \
  _macro(CastFromMirroredOp); \
  _macro(DistributeSplitOp);  \
  _macro(DistributeCloneOp);  \
  _macro(DistributeConcatOp); \
  _macro(DistributeAddOp);    \
  _macro(FunctionOp);

class NormalInterpreter : public OpExprInterpreter {
 public:
  NormalInterpreter() = delete;
  NormalInterpreter(const std::shared_ptr<OpExprInterpContext>& context)
      : OpExprInterpreter(), context_(context) {}
  virtual ~NormalInterpreter() = default;

  const OpExprInterpContext* context() const { return context_.get(); }

 private:
  std::shared_ptr<OpExprInterpContext> context_;
};

#define DECLARE_NORMAL_APPLY_FUNC(op_type)                                                         \
  virtual void Apply_(const op_type##Expr* op_expr, const TensorList& inputs, TensorList& outputs, \
                      const OpExprInterpState* state);

class LazyInterpreter : public NormalInterpreter {
 public:
  LazyInterpreter(const std::shared_ptr<OpExprInterpContext>& context)
      : NormalInterpreter(context) {}

  void Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprInterpState* state) override;

 private:
  DECLARE_NORMAL_APPLY_FUNC(BuiltinOp);
  DECLARE_NORMAL_APPLY_FUNC(FunctionOp);
};

class EagerInterpreter : public NormalInterpreter {
 public:
  EagerInterpreter(const std::shared_ptr<OpExprInterpContext>& context)
      : NormalInterpreter(context) {}

  void Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprInterpState* state) override;

 private:
  FOR_ALL_OPS(DECLARE_NORMAL_APPLY_FUNC);
};

#undef DECLARE_NORMAL_APPLY_FUNC
#undef FOR_ALL_OPS

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
