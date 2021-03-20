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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

class OpExprInterpState {
 public:
  const TensorTuple& SavedTensors() const { return saved_tensors_; }

  void SaveTensorForBackward(const std::shared_ptr<Tensor>& tensor) {
    saved_tensors_.push_back(tensor);
  }

 private:
  TensorTuple saved_tensors_;
};

class OpExprInterpreter {
 public:
  OpExprInterpreter() = default;
  virtual ~OpExprInterpreter() = default;

  virtual Maybe<OpExprInterpState> Apply(const OpExpr& op, const OpExprInterpState* state,
                                         const TensorTuple& inputs, TensorTuple* outputs) = 0;
};

#define FOR_EACH_OPS(_macro)  \
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
  NormalInterpreter() : OpExprInterpreter() {}
  virtual ~NormalInterpreter() = default;
};

#define DECLARE_NORMAL_APPLY_FUNC(op_type)                                   \
  virtual Maybe<OpExprInterpState> ApplyImpl(const op_type##Expr& op_expr,   \
                                             const OpExprInterpState* state, \
                                             const TensorTuple& inputs, TensorTuple* outputs);

class LazyInterpreter : public NormalInterpreter {
 public:
  LazyInterpreter() : NormalInterpreter() {}

  Maybe<OpExprInterpState> Apply(const OpExpr& op_expr, const OpExprInterpState* state,
                                 const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  DECLARE_NORMAL_APPLY_FUNC(BuiltinOp);
  DECLARE_NORMAL_APPLY_FUNC(FunctionOp);
};

class EagerInterpreter : public NormalInterpreter {
 public:
  EagerInterpreter() : NormalInterpreter() {}

  Maybe<OpExprInterpState> Apply(const OpExpr& op_expr, const OpExprInterpState* state,
                                 const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  FOR_EACH_OPS(DECLARE_NORMAL_APPLY_FUNC);
};

#undef DECLARE_NORMAL_APPLY_FUNC
#undef FOR_EACH_OPS

class AutogradInterpreter : public OpExprInterpreter {
 public:
  AutogradInterpreter() = delete;
  AutogradInterpreter(const std::shared_ptr<NormalInterpreter>& normal_interp)
      : OpExprInterpreter(), normal_interp_(normal_interp) {}

  Maybe<OpExprInterpState> Apply(const OpExpr& op_expr, const OpExprInterpState* state,
                                 const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  std::shared_ptr<NormalInterpreter> normal_interp_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_H_
