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
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/attr_value_map.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_conf.pb.h"

namespace oneflow {
namespace one {

class OpExprGradFunctionIf;
class OpExprGradClosure;

class OpExpr {
 public:
  virtual ~OpExpr() = default;
  virtual const std::string type_name() const = 0;

  virtual int input_size() const = 0;
  virtual int output_size() const = 0;

  virtual Maybe<bool> IsGradDisabled() const = 0;

  virtual Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const = 0;

 protected:
  OpExpr() = default;
};

class BuiltinOpExpr : public OpExpr {
 public:
  explicit BuiltinOpExpr(const std::string& op_name, const std::vector<std::string>& indexed_ibns,
                         const std::vector<std::string>& indexed_obns);

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }

  int input_size() const override { return indexed_ibns_.size(); }
  int output_size() const override { return indexed_obns_.size(); }

  const std::vector<std::string>& indexed_ibns() const { return indexed_ibns_; }
  const std::vector<std::string>& indexed_obns() const { return indexed_obns_; }
  const std::vector<std::pair<std::string, int32_t>>& indexed_input_pairs() const {
    return indexed_input_pairs_;
  }
  const std::vector<std::pair<std::string, int32_t>>& indexed_output_pairs() const {
    return indexed_output_pairs_;
  }

  virtual Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrValueMap& attrs) const = 0;

 protected:
  std::string op_name_;
  // The indexed input blob names.
  std::vector<std::string> indexed_ibns_;
  // The indexed output blob names.
  std::vector<std::string> indexed_obns_;
  std::vector<std::pair<std::string, int32_t>> indexed_input_pairs_;
  std::vector<std::pair<std::string, int32_t>> indexed_output_pairs_;
};

template<typename ProtoType>
class BuiltinOpExprImpl : public BuiltinOpExpr {
 public:
  explicit BuiltinOpExprImpl(const std::string& op_name, ProtoType&& op_proto,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
      : BuiltinOpExpr(op_name, indexed_ibns, indexed_obns), op_proto_(std::move(op_proto)) {}

  virtual ~BuiltinOpExprImpl() = default;

  const ProtoType& proto() const { return op_proto_; }
  ProtoType* mutable_proto() { return &op_proto_; }

  const std::string type_name() const override;

  Maybe<bool> IsGradDisabled() const override { return false; }

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

  Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrValueMap& attrs) const override;

 protected:
  ProtoType op_proto_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class StatefulOpKernel;

class UserOpExpr : public BuiltinOpExprImpl<UserOpConf> {
 public:
  UserOpExpr() = default;
  virtual ~UserOpExpr() = default;
  explicit UserOpExpr(const std::string& op_name, UserOpConf&& proto,
                      const std::vector<std::string>& indexed_ibns,
                      const std::vector<std::string>& indexed_obns)
      : BuiltinOpExprImpl<UserOpConf>(op_name, std::move(proto), indexed_ibns, indexed_obns){};

  Maybe<StatefulOpKernel> MutKernel4Device(const Device& device) const;

 private:
  mutable HashMap<Device, std::shared_ptr<StatefulOpKernel>> device2kernel_;
};

using VariableOpExpr = BuiltinOpExprImpl<VariableOpConf>;
using CastToMirroredOpExpr = BuiltinOpExprImpl<CastToMirroredOpConf>;
using CastFromMirroredOpExpr = BuiltinOpExprImpl<CastFromMirroredOpConf>;
using DistributeSplitOpExpr = BuiltinOpExprImpl<DistributeSplitOpConf>;
using DistributeCloneOpExpr = BuiltinOpExprImpl<DistributeCloneOpConf>;
using DistributeConcatOpExpr = BuiltinOpExprImpl<DistributeConcatOpConf>;
using DistributeAddOpExpr = BuiltinOpExprImpl<DistributeAddOpConf>;

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

  const std::string type_name() const override { return "function"; }

  int input_size() const override { UNIMPLEMENTED(); }
  int output_size() const override { UNIMPLEMENTED(); }

  FType forward() const { return forward_; }
  FType backward() const { return backward_; }

  std::shared_ptr<const OpExprInterpState> state() const { return state_; }
  std::shared_ptr<OpExprInterpState> mutable_state() { return state_; }

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override { UNIMPLEMENTED(); }

 private:
  FType forward_;
  FType backward_;
  std::shared_ptr<OpExprInterpState> state_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
