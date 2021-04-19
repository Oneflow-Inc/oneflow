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

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {
namespace one {

#define DEFINE_OPEXPR_TYPE_NAME(_T, _type_name)                \
  template<>                                                   \
  const std::string BuiltinOpExprImpl<_T>::type_name() const { \
    return _type_name;                                         \
  }

DEFINE_OPEXPR_TYPE_NAME(UserOpConf, "user");
DEFINE_OPEXPR_TYPE_NAME(VariableOpConf, "variable");
DEFINE_OPEXPR_TYPE_NAME(CastToMirroredOpConf, "cast_to_mirrored");
DEFINE_OPEXPR_TYPE_NAME(CastFromMirroredOpConf, "cast_from_mirrored");
DEFINE_OPEXPR_TYPE_NAME(DistributeSplitOpConf, "distribute_split");
DEFINE_OPEXPR_TYPE_NAME(DistributeCloneOpConf, "distribute_clone");
DEFINE_OPEXPR_TYPE_NAME(DistributeConcatOpConf, "distribute_concat");
DEFINE_OPEXPR_TYPE_NAME(DistributeAddOpConf, "distribute_add");

template<>
Maybe<void> BuiltinOpExprImpl<UserOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_user_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<UserOpConf>::IsGradDisabled() const {
  const std::string& op_type_name = op_proto_.op_type_name();
  const user_op::OpGradRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpGradRegistryResult(op_type_name);
  if (val) { return false; }
  return !IsClassRegistered<std::string, OpExprGradFunction>(op_type_name);
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<UserOpConf>::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    if (IsClassRegistered<std::string, OpExprGradFunction>(proto().op_type_name())) {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunction>(proto().op_type_name()));
    } else {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunction>("default"));
    }
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

template<>
Maybe<void> BuiltinOpExprImpl<VariableOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_variable_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<VariableOpConf>::IsGradDisabled() const {
  return true;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<VariableOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastToMirroredOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_to_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<CastToMirroredOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastToMirroredOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastFromMirroredOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_from_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<CastFromMirroredOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastFromMirroredOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeSplitOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_split_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<DistributeSplitOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeSplitOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeCloneOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_clone_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<DistributeCloneOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeCloneOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeConcatOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_concat_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<DistributeConcatOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeConcatOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeAddOpConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_add_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<bool> BuiltinOpExprImpl<DistributeAddOpConf>::IsGradDisabled() const {
  return false;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeAddOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace one
}  // namespace oneflow
