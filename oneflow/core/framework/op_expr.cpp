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
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace one {

namespace {
std::pair<std::string, int> GetPair(const std::string& bn) {
  const size_t pos = bn.rfind('_');
  CHECK_NE(pos, std::string::npos) << "bn: " << bn;
  return std::make_pair(bn.substr(0, pos), std::stoi(bn.substr(pos + 1)));
};
}  // namespace

BuiltinOpExpr::BuiltinOpExpr(const std::string& type, const std::string& op_name,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
    : OpExpr(type), op_name_(op_name), indexed_ibns_(indexed_ibns), indexed_obns_(indexed_obns) {
  for (const auto& ibn : indexed_ibns) { indexed_input_pairs_.push_back(GetPair(ibn)); }
  for (const auto& obn : indexed_obns) { indexed_output_pairs_.push_back(GetPair(obn)); }
}

UserOpExpr::UserOpExpr(const std::string& op_name, UserOpConf&& proto,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns)
    : BuiltinOpExpr("user", op_name, indexed_ibns, indexed_obns), proto_(std::move(proto)) {
  OperatorConf op_conf;
  BuildOpConf(&op_conf, {});
  // TODO: align with pytorch: set device tag in Interpret according to inputs
  op_conf.set_device_tag("gpu");
  auto mem_case = MemoryCaseUtil::MakeMemCase(DeviceType::kGPU, 0);
  kernel_ = std::make_shared<StatefulOpKernel>(op_conf, mem_case, &indexed_input_pairs(),
                                               &indexed_output_pairs());
}

Maybe<void> UserOpExpr::BuildOpConf(OperatorConf* op_conf, const AttrValueMap& attrs) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_user_conf()) = proto_;
  auto* user_op_conf = op_conf->mutable_user_conf();
  for (const auto& it : attrs) {
    AttrValue attr_val;
    it.second->ToProto(&attr_val);
    (*(user_op_conf->mutable_attr()))[it.first] = attr_val;
  }
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> UserOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    if (IsClassRegistered<std::string, OpExprGradFunction>(proto().op_type_name())) {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunction>(proto().op_type_name()));
    } else {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunction>("default"));
    }
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    op_grad_func_->Init(*this);
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

Maybe<void> VariableOpExpr::BuildOpConf(OperatorConf* op_conf, const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_variable_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> VariableOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> CastToMirroredOpExpr::BuildOpConf(OperatorConf* op_conf,
                                              const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_to_mirrored_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> CastToMirroredOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> CastFromMirroredOpExpr::BuildOpConf(OperatorConf* op_conf,
                                                const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_from_mirrored_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> CastFromMirroredOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> DistributeSplitOpExpr::BuildOpConf(OperatorConf* op_conf,
                                               const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_split_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> DistributeSplitOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> DistributeCloneOpExpr::BuildOpConf(OperatorConf* op_conf,
                                               const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_clone_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> DistributeCloneOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> DistributeConcatOpExpr::BuildOpConf(OperatorConf* op_conf,
                                                const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_concat_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> DistributeConcatOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<void> DistributeAddOpExpr::BuildOpConf(OperatorConf* op_conf,
                                             const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_add_conf()) = proto_;
  return Maybe<void>::Ok();
}

Maybe<OpExprGradClosure> DistributeAddOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace one
}  // namespace oneflow
