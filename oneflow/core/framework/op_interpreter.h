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

  virtual Maybe<void> Apply(const OpExpr& op, const TensorTuple& inputs,
                            TensorTuple* outputs) const = 0;
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

#define DECLARE_NORMAL_APPLY_FUNC(op_type)                                               \
  virtual Maybe<void> ApplyImpl(const op_type##Expr& op_expr, const TensorTuple& inputs, \
                                TensorTuple* outputs) const;

class LazyInterpreter : public OpExprInterpreter {
 public:
  LazyInterpreter() : OpExprInterpreter() {}

  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                    TensorTuple* outputs) const override;

 private:
  DECLARE_NORMAL_APPLY_FUNC(BuiltinOp);
  DECLARE_NORMAL_APPLY_FUNC(FunctionOp);
};

class EagerInterpreter : public OpExprInterpreter {
 public:
  EagerInterpreter() : OpExprInterpreter() {}

  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                    TensorTuple* outputs) const override;

 private:
  FOR_EACH_OPS(DECLARE_NORMAL_APPLY_FUNC);
};

#undef DECLARE_NORMAL_APPLY_FUNC
#undef FOR_EACH_OPS

class AutogradInterpreter {
 public:
  AutogradInterpreter() = delete;
  AutogradInterpreter(const std::shared_ptr<OpExprInterpreter>& internal) : internal_(internal) {}

  virtual ~AutogradInterpreter() = default;

  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs, TensorTuple* outputs) const;

 private:
  std::shared_ptr<OpExprInterpreter> internal_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_H_
