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
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace one {

class OpExpr {
 public:
  explicit OpExpr(const std::string& type) : type_(type) {}
  virtual ~OpExpr() = default;

  const std::string& type() const { return type_; }

  virtual int input_num() const = 0;
  virtual int output_num() const = 0;

 private:
  std::string type_;
};

class BuiltinOpExpr : public OpExpr {
 public:
  explicit BuiltinOpExpr(const std::string& type, const std::string& op_name,
                         const std::vector<std::string>& indexed_ibns,
                         const std::vector<std::string>& indexed_obns)
      : OpExpr(type), op_name_(op_name), indexed_ibns_(indexed_ibns), indexed_obns_(indexed_obns) {}

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }

  int input_num() const override { return indexed_ibns_.size(); }
  int output_num() const override { return indexed_obns_.size(); }

  const std::vector<std::string>& indexed_ibns() const { return indexed_ibns_; }
  const std::vector<std::string>& indexed_obns() const { return indexed_obns_; }

  virtual void BuildOpConf(OperatorConf* op_conf) const = 0;

 protected:
  std::string op_name_;
  // The indexed input blob names.
  std::vector<std::string> indexed_ibns_;
  // The indexed output blob names.
  std::vector<std::string> indexed_obns_;
};

#define DEFINE_BUILTIN_OPEXPR_CLASS(_op_name, _op_conf)                                  \
  class _op_name##Expr : public BuiltinOpExpr {                                          \
   public:                                                                               \
    _op_name##Expr() = default;                                                          \
    virtual ~_op_name##Expr() = default;                                                 \
    explicit _op_name##Expr(const std::string& op_name, _op_name##Conf&& proto,          \
                            const std::vector<std::string>& indexed_ibns,                \
                            const std::vector<std::string>& indexed_obns)                \
        : BuiltinOpExpr(OF_PP_STRINGIZE(_op_name), op_name, indexed_ibns, indexed_obns), \
          proto_(std::move(proto)) {}                                                    \
                                                                                         \
    const _op_name##Conf& proto() const { return proto_; }                               \
    _op_name##Conf* mutable_proto() { return &proto_; }                                  \
                                                                                         \
    void BuildOpConf(OperatorConf* op_conf) const {                                      \
      *(op_conf->mutable_name()) = this->op_name_;                                       \
      *(op_conf->mutable_##_op_conf##_conf()) = proto_;                                  \
    }                                                                                    \
                                                                                         \
   private:                                                                              \
    _op_name##Conf proto_;                                                               \
  };

DEFINE_BUILTIN_OPEXPR_CLASS(UserOp, user);
DEFINE_BUILTIN_OPEXPR_CLASS(VariableOp, variable);
DEFINE_BUILTIN_OPEXPR_CLASS(CastToMirroredOp, cast_to_mirrored);
DEFINE_BUILTIN_OPEXPR_CLASS(CastFromMirroredOp, cast_from_mirrored);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeSplitOp, distribute_split);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeCloneOp, distribute_clone);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeConcatOp, distribute_concat);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeAddOp, distribute_add);

#undef DEFINE_BUILTIN_OPEXPR_CLASS

// TODO(): Finish the class definition of `FunctionOpExpr`.
class FunctionOpExpr : public OpExpr {
 public:
  FunctionOpExpr() : OpExpr("FunctionOp") {}
  virtual ~FunctionOpExpr() = default;

  int input_num() const override { UNIMPLEMENTED(); }
  int output_num() const override { UNIMPLEMENTED(); }
};

}  // namespace one
}  // namespace oneflow

#undef DEFINE_DEFAULT_CONSTRUCTOR

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
