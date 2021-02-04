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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_

#include "oneflow/core/framework/user_op_conf.pb.h"

namespace oneflow {
namespace one {

class Tensor {};
using TensorRef = std::shared_ptr<Tensor>;
using TensorList = std::vector<TensorRef>;

class OpExprInterpreter;
class OpExprEvalState;

class OpExpr {
 public:
  virtual void evaluate(OpExprInterpreter* evaluator, const TensorList& inputs, TensorList& outputs,
                        const OpExprEvalState* state) = 0;
  // TODO(): Uncomment.
  // virtual FilterInputTensorsUsedByBackward(const TensorList& inputs) = 0;
  // virtual FilterOutputTensorsUsedByBackward(const TensorList& outputs) = 0;

  virtual std::shared_ptr<OpExpr> GetBackwardOpExpr() = 0;
};

class BuiltinOpExpr : public OpExpr {
 public:
  BuiltinOpExpr() = default;
  explicit BuiltinOpExpr(const std::string& op_name) : op_name_(op_name) {}

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }
  void set_op_name(const std::string& op_name) { op_name_ = op_name; }

 private:
  // The operation name.
  std::string op_name_;
};

class UserOpExpr : public BuiltinOpExpr {
 public:
  UserOpExpr() = default;
  explicit UserOpExpr(const std::string& op_name) : BuiltinOpExpr(op_name) {}

  void evaluate(OpExprInterpreter* evaluator, const TensorList& inputs, TensorList& outputs,
                const OpExprEvalState* state) override;

  std::shared_ptr<OpExpr> GetBackwardOpExpr() override;

  const UserOpConf& proto() const { return proto_; }
  UserOpConf* mutable_proto() { return &proto_; }

  const std::vector<std::string>& indexed_input_names() const { return indexed_input_names_; }

  friend class OpBuilder;

 private:
  // The internal operation proto.
  UserOpConf proto_;

  // The indexed input operand names.
  std::vector<std::string> indexed_input_names_;
};

class OpExprEvalState {
 public:
  OpExprEvalState() = default;
  virtual ~OpExprEvalState() = default;

  const TensorList& SavedTensors() const { return saved_tensors_; }

  void SaveTensorForBackward(const TensorRef& tensor) { saved_tensors_.push_back(tensor); }

 private:
  TensorList saved_tensors_;
};

class OpExprInterpreter {
 public:
  OpExprInterpreter() : self_state_(new OpExprEvalState) {}
  virtual ~OpExprInterpreter() = default;

  virtual void apply(const OpExpr* op, const TensorList& inputs, TensorList& outputs,
                     const OpExprEvalState* state) = 0;

  std::shared_ptr<OpExprEvalState> state() const { return self_state_; }

 protected:
  std::shared_ptr<OpExprEvalState> self_state_;
};

class NormalInterpreter : public OpExprInterpreter {
 public:
  NormalInterpreter() : OpExprInterpreter() {}

  void apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprEvalState* state) override;
};

class AutogradInterpreter : public OpExprInterpreter {
 public:
  AutogradInterpreter() : OpExprInterpreter() {}

  void apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
             const OpExprEvalState* state) override;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
