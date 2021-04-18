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

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_type_trait.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace one {

class OpExprGradFunction;
class OpExprGradClosure;

class OpExpr {
 public:
  virtual ~OpExpr() = default;
  virtual const std::string type() const = 0;

  virtual int input_size() const = 0;
  virtual int output_size() const = 0;

  virtual bool IsGradDisabled() const { return false; }

  virtual Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const = 0;

 protected:
  OpExpr() = default;
};

class BuiltinOpExpr : public OpExpr {
 public:
  explicit BuiltinOpExpr(const std::string& op_name, const std::vector<std::string>& indexed_ibns,
                         const std::vector<std::string>& indexed_obns)
      : op_name_(op_name), indexed_ibns_(indexed_ibns), indexed_obns_(indexed_obns) {}

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }

  int input_size() const override { return indexed_ibns_.size(); }
  int output_size() const override { return indexed_obns_.size(); }

  const std::vector<std::string>& indexed_ibns() const { return indexed_ibns_; }
  const std::vector<std::string>& indexed_obns() const { return indexed_obns_; }

  virtual Maybe<void> BuildOpConf(OperatorConf* op_conf) const = 0;

 protected:
  std::string op_name_;
  // The indexed input blob names.
  std::vector<std::string> indexed_ibns_;
  // The indexed output blob names.
  std::vector<std::string> indexed_obns_;

  mutable std::shared_ptr<OpExprGradFunction> op_grad_func_;
};

template<OperatorConf::OpTypeCase op_type_case>
class BuiltinOpExprImpl : public BuiltinOpExpr {
 public:
  using proto_type = typename OpTypeTrait<op_type_case>::proto_type;
  explicit BuiltinOpExprImpl(const std::string& op_name, proto_type&& op_proto,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
      : BuiltinOpExpr(op_name, indexed_ibns, indexed_obns), op_proto_(std::move(op_proto)) {}

  virtual ~BuiltinOpExprImpl() = default;

  const proto_type& proto() const { return op_proto_; }
  proto_type* mutable_proto() { return &op_proto_; }

  Maybe<void> BuildOpConf(OperatorConf* op_conf) const override;

  const std::string type() const override { return OpTypeTrait<op_type_case>::op_type_name(); }

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 protected:
  proto_type op_proto_;
};

using UserOpExpr = BuiltinOpExprImpl<OperatorConf::kUserConf>;
using VariableOpExpr = BuiltinOpExprImpl<OperatorConf::kVariableConf>;
using CastToMirroredOpExpr = BuiltinOpExprImpl<OperatorConf::kCastToMirroredConf>;
using CastFromMirroredOpExpr = BuiltinOpExprImpl<OperatorConf::kCastFromMirroredConf>;
using DistributeSplitOpExpr = BuiltinOpExprImpl<OperatorConf::kDistributeSplitConf>;
using DistributeCloneOpExpr = BuiltinOpExprImpl<OperatorConf::kDistributeCloneConf>;
using DistributeConcatOpExpr = BuiltinOpExprImpl<OperatorConf::kDistributeConcatConf>;
using DistributeAddOpExpr = BuiltinOpExprImpl<OperatorConf::kDistributeAddConf>;

class OpExprInterpState;
// TODO(): Finish the class definition of `FunctionOpExpr`.
class FunctionOpExpr : public OpExpr {
 public:
  using FType = std::function<Maybe<void>(const std::shared_ptr<OpExprInterpState>& /*ctx*/,
                                          const TensorTuple& /*inputs or out_grads*/,
                                          TensorTuple* /*outputs or in_grads*/)>;

  FunctionOpExpr(const FType& forward, const FType& backward)
      : OpExpr(), forward_(forward), backward_(backward) {}
  virtual ~FunctionOpExpr() = default;

  const std::string type() const override { return "function"; }

  int input_size() const override { UNIMPLEMENTED(); }
  int output_size() const override { UNIMPLEMENTED(); }

  FType forward() const { return forward_; }
  FType backward() const { return backward_; }

  std::shared_ptr<const OpExprInterpState> state() const { return state_; }
  std::shared_ptr<OpExprInterpState> mutable_state() { return state_; }

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override { UNIMPLEMENTED(); }

 private:
  FType forward_;
  FType backward_;
  std::shared_ptr<OpExprInterpState> state_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
