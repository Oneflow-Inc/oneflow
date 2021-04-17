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

namespace oneflow {
namespace one {

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kUserConf>::BuildOpConf(OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_user_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<OperatorConf::kUserConf>::GetOrCreateOpGradClosure()
    const {
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
Maybe<void> BuiltinOpExprImpl<OperatorConf::kVariableConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_variable_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<OperatorConf::kVariableConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kCastToMirroredConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_to_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kCastToMirroredConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kCastFromMirroredConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_from_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kCastFromMirroredConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kDistributeSplitConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_split_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kDistributeSplitConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kDistributeCloneConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_clone_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kDistributeCloneConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kDistributeConcatConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_concat_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kDistributeConcatConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<OperatorConf::kDistributeAddConf>::BuildOpConf(
    OperatorConf* op_conf) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_add_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<OperatorConf::kDistributeAddConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace one
}  // namespace oneflow
