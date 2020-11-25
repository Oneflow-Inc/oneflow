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
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

namespace user_op {

UserOpConfWrapper::UserOpConfWrapper(const OperatorConf& op_conf) : op_conf_(op_conf) {
  CHECK(op_conf_.has_user_conf());
  for (const auto& kv : op_conf_.user_conf().attr()) {
    AttrValue::ValueCase value_case = kv.second.value_case();
    switch (value_case) {
#define CASE_ENTRY(field, cpp_type, attr_type)                                      \
  /* AttrValue::ValueCase has the same order and naming convention as AttrType */   \
  case (static_cast<AttrValue::ValueCase>(attr_type)):                              \
    CHECK(attrs_                                                                    \
              .emplace(kv.first, std::make_shared<TypedAttrVal<cpp_type>>(          \
                                     AttrValueAccessor<cpp_type>::Attr(kv.second))) \
              .second);                                                             \
    break;
      OF_PP_FOR_EACH_TUPLE(CASE_ENTRY, ATTR_SEQ)
#undef CASE_ENTRY
      default: LOG(FATAL) << "Wrong attr value type: " << static_cast<int32_t>(value_case);
    };
  }
}

const OperatorConf& UserOpConfWrapper::op_conf() const { return op_conf_; }

const UserOpConf& UserOpConfWrapper::user_op_conf() const { return op_conf_.user_conf(); }

const std::string& UserOpConfWrapper::op_name() const { return op_conf_.name(); }

const std::string& UserOpConfWrapper::op_type_name() const {
  return op_conf_.user_conf().op_type_name();
}

const std::string& UserOpConfWrapper::input(const std::string& arg_name, int32_t index) const {
  auto it = op_conf_.user_conf().input().find(arg_name);
  CHECK(it != op_conf_.user_conf().input().end())
      << "arg_name: " << arg_name << ", index: " << index;
  CHECK(index >= 0 && index < it->second.s_size());
  return it->second.s(index);
}

const std::string& UserOpConfWrapper::output(const std::string& arg_name, int32_t index) const {
  auto it = op_conf_.user_conf().output().find(arg_name);
  CHECK(it != op_conf_.user_conf().output().end())
      << "arg_name: " << arg_name << ", index: " << index;
  CHECK(index >= 0 && index < it->second.s_size());
  return it->second.s(index);
}

bool UserOpConfWrapper::has_input(const std::string& arg_name, int32_t index) const {
  return input_size(arg_name) > index;
}

bool UserOpConfWrapper::has_output(const std::string& arg_name, int32_t index) const {
  return output_size(arg_name) > index;
}

int32_t UserOpConfWrapper::input_size(const std::string& arg_name) const {
  auto it = op_conf_.user_conf().input().find(arg_name);
  if (it == op_conf_.user_conf().input().end()) { return 0; }
  return it->second.s_size();
}

int32_t UserOpConfWrapper::output_size(const std::string& arg_name) const {
  auto it = op_conf_.user_conf().output().find(arg_name);
  if (it == op_conf_.user_conf().output().end()) { return 0; }
  return it->second.s_size();
}

#define OP_WRAPPER_ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                    \
  template<>                                                                                       \
  const cpp_type& UserOpConfWrapper::attr<cpp_type>(const std::string& attr_name) const {          \
    auto it = attrs_.find(attr_name);                                                              \
    if (it != attrs_.end()) {                                                                      \
      return std::dynamic_pointer_cast<TypedAttrVal<cpp_type>>(it->second)->val();                 \
    } else {                                                                                       \
      LOG(FATAL) << "Cannot find the attr: " << attr_name                                          \
                 << " with AttrType: " << static_cast<int32_t>(attr_type);                         \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  template<>                                                                                       \
  UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Attr<cpp_type>(const std::string& attr_name, \
                                                                     const cpp_type& val) {        \
    AttrValue attr_val;                                                                            \
    AttrValueAccessor<cpp_type>::Attr(val, &attr_val);                                             \
    attr_.emplace(attr_name, attr_val);                                                            \
    return *this;                                                                                  \
  }

OF_PP_FOR_EACH_TUPLE(OP_WRAPPER_ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef OP_WRAPPER_ATTR_MEMBER_FUNC

UserOpWrapper::UserOpWrapper(
    const OperatorConf& op,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp)
    : conf_(op), diff_fn_(DiffLbi4BnInOp) {
  auto InitTensorDescFromOpArgs = [&](const PbMap<std::string, UserOpConf_ListString>& args) {
    for (const auto& pair : args) {
      for (int32_t i = 0; i < pair.second.s_size(); ++i) {
        std::string bn = GenRepeatedBn(pair.first, i);
        const BlobDesc& blob_desc = LogicalBlobDesc4BnInOp(bn);
        CHECK((&blob_desc) != nullptr);
        BlobDescProto proto;
        blob_desc.ToProto(&proto);
        TensorDesc tensor_desc(proto);
        CHECK(bn2tensor_desc_.emplace(bn, tensor_desc).second);
      }
    }
  };
  InitTensorDescFromOpArgs(op.user_conf().input());
  InitTensorDescFromOpArgs(op.user_conf().output());
}

bool UserOpWrapper::NeedGenGradTensor4OpInput(const std::string& input_arg_name,
                                              int32_t index) const {
  auto it = op_conf().user_conf().input().find(input_arg_name);
  CHECK(it != op_conf().user_conf().input().end())
      << "arg_name: " << input_arg_name << ", index: " << index;
  CHECK(index >= 0 && index < it->second.s_size())
      << "arg_name: " << input_arg_name << ", index: " << index;
  return diff_fn_(GenRepeatedBn(input_arg_name, index)) != nullptr;
}

bool UserOpWrapper::HasGradTensor4OpOutput(const std::string& output_arg_name,
                                           int32_t index) const {
  auto it = op_conf().user_conf().output().find(output_arg_name);
  CHECK(it != op_conf().user_conf().output().end())
      << "arg_name: " << output_arg_name << ", index: " << index;
  CHECK(index >= 0 && index < it->second.s_size())
      << "arg_name: " << output_arg_name << ", index: " << index;
  return diff_fn_(GenRepeatedBn(output_arg_name, index)) != nullptr;
}

std::string UserOpWrapper::output_grad(const std::string& output_arg_name, int32_t index) const {
  auto it = op_conf().user_conf().output().find(output_arg_name);
  CHECK(it != op_conf().user_conf().output().end())
      << "arg_name: " << output_arg_name << ", index: " << index;
  CHECK(index >= 0 && index < it->second.s_size())
      << "arg_name: " << output_arg_name << ", index: " << index;
  return GenLogicalBlobName(*diff_fn_(GenRepeatedBn(output_arg_name, index)));
}

std::string UserOpWrapper::GetGradTensorWithOpOutput(const std::string& output_arg_name,
                                                     int32_t index) const {
  return output_grad(output_arg_name, index);
}

void UserOpWrapper::BindGradTensorWithOpInput(const std::string& logical_grad_blob_name,
                                              const std::string& input_arg_name,
                                              int32_t index) const {
  CHECK(NeedGenGradTensor4OpInput(input_arg_name, index));
  *diff_fn_(GenRepeatedBn(input_arg_name, index)) = GenLogicalBlobId(logical_grad_blob_name);
}

const TensorDesc& UserOpWrapper::arg_tensor_desc(const std::string& arg_name, int32_t index) const {
  std::string bn = GenRepeatedBn(arg_name, index);
  CHECK(bn2tensor_desc_.find(bn) != bn2tensor_desc_.end());
  return bn2tensor_desc_.at(bn);
}

const TensorDesc& UserOpWrapper::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                            int32_t index) const {
  return arg_tensor_desc(arg_name, index);
}

void UserOpWrapper::InputGradBind(const user_op::OpArg& input,
                                  const UserOpInputGradGetFn& grad_fn) {
  if (NeedGenGradTensor4OpInput(input.name(), input.index())) {
    BindGradTensorWithOpInput(grad_fn(), input.name(), input.index());
  }
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::InputBind(
    const std::string& arg_name, const std::string& logical_blob_name) {
  input_[arg_name].push_back(logical_blob_name);
  return *this;
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Input(const std::string& arg_name,
                                                          const std::string& logical_blob_name) {
  return InputBind(arg_name, logical_blob_name);
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Output(const std::string& arg_name) {
  return Output(arg_name, 1);
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Output(const std::string& arg_name,
                                                           int32_t num) {
  CHECK(num >= 0);
  output_[arg_name].resize(num);
  for (int32_t i = 0; i < num; ++i) {
    std::string bn = GenRepeatedBn(arg_name, i);
    output_[arg_name].at(i) = GenLogicalBlobName(op_name_, bn);
  }
  return *this;
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::ScopeSymbolId(int64_t scope_symbol_id) {
  scope_symbol_id_.set_value(scope_symbol_id);
  return *this;
}

UserOpConfWrapper UserOpConfWrapperBuilder::Build() {
  OperatorConf op_conf;
  op_conf.set_name(op_name_);
  if (scope_symbol_id_.has_value()) { op_conf.set_scope_symbol_id(scope_symbol_id_.value()); }
  UserOpConf* user_conf = op_conf.mutable_user_conf();
  user_conf->set_op_type_name(op_type_name_);
  auto GenArgs = [&](const HashMap<std::string, std::vector<std::string>>& src,
                     PbMap<std::string, UserOpConf_ListString>* arg_name2lbns) {
    for (const auto& pair : src) {
      *(*arg_name2lbns)[pair.first].mutable_s() = StdVec2PbRpf<std::string>(pair.second);
    }
  };
  GenArgs(input_, user_conf->mutable_input());
  GenArgs(output_, user_conf->mutable_output());
  for (const auto& pair : attr_) { (*user_conf->mutable_attr())[pair.first] = pair.second; }
  wrapper_ = UserOpConfWrapper(*CHECK_JUST(CheckAndCompleteUserOpConfImpl(op_conf)));
  return wrapper_;
}

void BackwardOpConfContext::DefineOp(const std::string& op_name, const BackwardOpBuilderFn& fn) {
  auto it = op_builder_fns_.find(op_name);
  CHECK(it == op_builder_fns_.end()) << " op_name " << op_name << " has been defined.";
  op_builder_fns_[op_name] = fn;
}

UserOpConfWrapper& BackwardOpConfContext::GetOp(const std::string& op_name) {
  auto it = op_builder_results_.find(op_name);
  if (it != op_builder_results_.end()) {
    // return result from cache
    return it->second;
  } else {
    // build and put result into cache
    auto fn_it = op_builder_fns_.find(op_name);
    CHECK(fn_it != op_builder_fns_.end()) << " op_name " << op_name << " has no builder function.";
    CHECK(fn_it->second != nullptr) << " op_name " << op_name << " builder function is null.";
    UserOpConfWrapperBuilder builder(op_name);
    auto ret =
        op_builder_results_.emplace(std::make_pair(op_name, std::move(fn_it->second(builder))));
    CHECK(ret.second == true) << " op_name " << op_name << " build result insert failed.";

    // add new op conf
    bw_op_confs_->push_back(ret.first->second.op_conf());

    return ret.first->second;
  }
}

}  // namespace user_op

Maybe<void> CheckArgDefIsValidInUserOpConf(
    const OperatorConf& op_conf, const PbMap<std::string, UserOpConf_ListString>& arg_name2lbns,
    const PbRpf<UserOpDef_ArgDef>& args) {
  const std::string& op_name = op_conf.name();
  const std::string& op_type_name = op_conf.user_conf().op_type_name();
  HashSet<std::string> op_def_arg_names;
  for (const auto& arg : args) {
    int32_t arg_blob_num = 0;
    if (arg_name2lbns.find(arg.name()) != arg_name2lbns.end()) {
      arg_blob_num = arg_name2lbns.at(arg.name()).s_size();
    }
    if (arg_blob_num != arg.num()) {
      if (arg_blob_num == 0) {
        CHECK_OR_RETURN(arg.is_optional())
            << " op_name: " << op_name << " op_type_name: " << op_type_name
            << " arg name: " << arg.name() << " in OpDef must have blob in op_conf";
      } else {
        CHECK_OR_RETURN(arg_blob_num > arg.num() && arg.num_as_min())
            << " op_name: " << op_name << " op_type_name: " << op_type_name
            << " arg name: " << arg.name() << " has blob num: " << arg_blob_num
            << " in op_conf does not meet its constraints in OpDef";
      }
    }
    op_def_arg_names.insert(arg.name());
  }
  for (const auto& pair : arg_name2lbns) {
    CHECK_OR_RETURN(op_def_arg_names.find(pair.first) != op_def_arg_names.end())
        << " op_name: " << op_name << " op_type_name: " << op_type_name
        << " has not arg name: " << pair.first << " in OpDef";
  }
  return Maybe<void>::Ok();
}

Maybe<void> AddAttrDefaultValueAndCheckValid(const UserOpDef& op_def, OperatorConf* op_conf) {
  UserOpConf* user_conf = op_conf->mutable_user_conf();
  auto* attr_name2attr = user_conf->mutable_attr();
  HashSet<std::string> op_def_attr_names;
  for (const auto& attr : op_def.attr()) {
    if (attr_name2attr->find(attr.name()) == attr_name2attr->end()) {
      CHECK_OR_RETURN(attr.has_default_val())
          << " op_name: " << op_conf->name() << " op_type_name: " << user_conf->op_type_name()
          << " must set attr val for attr_name: " << attr.name();
      (*attr_name2attr)[attr.name()] = attr.default_val();
    }
    op_def_attr_names.insert(attr.name());
  }
  for (const auto& pair : user_conf->attr()) {
    CHECK_OR_RETURN(op_def_attr_names.find(pair.first) != op_def_attr_names.end())
        << " op_name: " << op_conf->name() << " op_type_name: " << user_conf->op_type_name()
        << " has not attr_name: " << pair.first << " in OpDef";
  }
  for (const auto& attr : op_def.attr()) {
    CHECK_OR_RETURN(static_cast<int32_t>(attr.type())
                    == static_cast<int32_t>(attr_name2attr->at(attr.name()).value_case()))
        << " op_name: " << op_conf->name() << " op_type_name: " << user_conf->op_type_name()
        << " attr_name: " << attr.name()
        << " has different attr type in OpDef and OpConf, it should be with type: "
        << AttrType_Name(attr.type());
  }
  return Maybe<void>::Ok();
}

Maybe<void> AddUserOpConfOutputDefaultArg(const UserOpDef& op_def, OperatorConf* op_conf) {
  UserOpConf* user_conf = op_conf->mutable_user_conf();
  // add default output arg and lbn
  for (const auto& output_arg : op_def.output()) {
    if (user_conf->output().find(output_arg.name()) == user_conf->output().end()
        && (!output_arg.is_optional()) && (!output_arg.num_as_min())) {
      for (int32_t i = 0; i < output_arg.num(); ++i) {
        std::string lbn = GenLogicalBlobName(op_conf->name(), GenRepeatedBn(output_arg.name(), i));
        (*(user_conf->mutable_output()))[output_arg.name()].add_s(lbn);
        CHECK_EQ(i + 1, user_conf->output().at(output_arg.name()).s_size());
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<long long> GetAttrTypeImpl(const std::string& op_type_name, const std::string& attr_name) {
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK_OR_RETURN(val) << " Cannot find op " << op_type_name;
  const UserOpDef& op_def = val->op_def;
  for (int32_t i = 0; i < op_def.attr_size(); ++i) {
    if (op_def.attr(i).name() == attr_name) { return op_def.attr(i).type(); }
  }
  CHECK_OR_RETURN(false) << " Cannot find attr " << attr_name << " in op " << op_type_name;
}

Maybe<OperatorConf> CheckAndCompleteUserOpConfImpl(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_user_conf()) << " Add default value only for user op";
  OperatorConf ret = op_conf;
  UserOpConf* user_conf = ret.mutable_user_conf();
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(user_conf->op_type_name());
  CHECK_OR_RETURN(val) << " Cannot find op_type_name: " << user_conf->op_type_name();
  const UserOpDef& op_def = val->op_def;

  JUST(AddAttrDefaultValueAndCheckValid(op_def, &ret));
  JUST(AddUserOpConfOutputDefaultArg(op_def, &ret));
  // check input and output valid
  JUST(CheckArgDefIsValidInUserOpConf(op_conf, user_conf->input(), op_def.input()));
  JUST(CheckArgDefIsValidInUserOpConf(op_conf, user_conf->output(), op_def.output()));
  // check attr valid by user
  JUST(val->check_fn(user_op::UserOpDefWrapper(op_def), user_op::UserOpConfWrapper(ret)));
  return ret;
}

}  // namespace oneflow
