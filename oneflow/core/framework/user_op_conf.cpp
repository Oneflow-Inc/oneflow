#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_attr.h"

namespace oneflow {

namespace user_op {

UserOpConfWrapper::UserOpConfWrapper(const OperatorConf& op_conf) {
  CHECK(op_conf.has_user_conf());
  op_conf_ = *CHECK_JUST(CheckAndCompleteUserOpConfImpl(op_conf));
}

const OperatorConf& UserOpConfWrapper::op_conf() const { return op_conf_; }

const std::string& UserOpConfWrapper::op_name() const { return op_conf_.name(); }

const std::string& UserOpConfWrapper::op_type_name() const {
  return op_conf_.user_conf().op_type_name();
}

const std::string& UserOpConfWrapper::input(const std::string& arg_name, int32_t index) const {
  auto it = op_conf_.user_conf().input().find(arg_name);
  CHECK(it != op_conf_.user_conf().input().end());
  CHECK(index >= 0 && index < it->second.s_size());
  return it->second.s(index);
}

const std::string& UserOpConfWrapper::output(const std::string& arg_name, int32_t index) const {
  auto it = op_conf_.user_conf().output().find(arg_name);
  CHECK(it != op_conf_.user_conf().output().end());
  CHECK(index >= 0 && index < it->second.s_size());
  return it->second.s(index);
}

#define OP_WRAPPER_ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                    \
  template<>                                                                                       \
  cpp_type UserOpConfWrapper::attr<cpp_type>(const std::string& attr_name) const {                 \
    CHECK(op_conf_.user_conf().attr().find(attr_name) != op_conf_.user_conf().attr().end());       \
    UserOpAttrVal val = op_conf_.user_conf().attr().at(attr_name);                                 \
    CHECK(val.has_##field());                                                                      \
    return val.field();                                                                            \
  }                                                                                                \
                                                                                                   \
  template<>                                                                                       \
  UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Attr<cpp_type>(const std::string& attr_name, \
                                                                     const cpp_type& val) {        \
    UserOpAttrVal attr_val;                                                                        \
    attr_val.set_##field(val);                                                                     \
    attr_.emplace(attr_name, attr_val);                                                            \
    return *this;                                                                                  \
  }

OF_PP_FOR_EACH_TUPLE(OP_WRAPPER_ATTR_MEMBER_FUNC, BASIC_ATTR_SEQ)

#undef OP_WRAPPER_ATTR_MEMBER_FUNC

template<>
Shape UserOpConfWrapper::attr<Shape>(const std::string& attr_name) const {
  CHECK(op_conf_.user_conf().attr().find(attr_name) != op_conf_.user_conf().attr().end());
  UserOpAttrVal val = op_conf_.user_conf().attr().at(attr_name);
  CHECK(val.has_at_shape());
  return Shape(val.at_shape());
}

template<>
UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Attr<Shape>(const std::string& attr_name,
                                                                const Shape& val) {
  UserOpAttrVal attr_val;
  val.ToProto(attr_val.mutable_at_shape());
  attr_.emplace(attr_name, attr_val);
  return *this;
}

#define OP_WRAPPER_LIST_ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                               \
  template<>                                                                                       \
  cpp_type UserOpConfWrapper::attr<cpp_type>(const std::string& attr_name) const {                 \
    CHECK(op_conf_.user_conf().attr().find(attr_name) != op_conf_.user_conf().attr().end());       \
    UserOpAttrVal val = op_conf_.user_conf().attr().at(attr_name);                                 \
    CHECK(val.has_##field());                                                                      \
    return PbRf2StdVec<cpp_type::value_type>(val.field().val());                                   \
  }                                                                                                \
                                                                                                   \
  template<>                                                                                       \
  UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Attr<cpp_type>(const std::string& attr_name, \
                                                                     const cpp_type& val) {        \
    UserOpAttrVal attr_val;                                                                        \
    *(attr_val.mutable_##field()->mutable_val()) = StdVec2PbRf<cpp_type::value_type>(val);         \
    attr_.emplace(attr_name, attr_val);                                                            \
    return *this;                                                                                  \
  }

OF_PP_FOR_EACH_TUPLE(OP_WRAPPER_LIST_ATTR_MEMBER_FUNC, LIST_ATTR_SEQ)

#undef OP_WRAPPER_LIST_ATTR_MEMBER_FUNC

UserOpWrapper::UserOpWrapper(
    const OperatorConf& op,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp)
    : conf_(op), diff_fn_(DiffLbi4BnInOp) {
  auto InitBlobDefFromOpArgs = [&](const PbMap<std::string, UserOpConf_ListString>& args) {
    for (const auto& pair : args) {
      for (int32_t i = 0; i < pair.second.s_size(); ++i) {
        std::string bn = GenRepeatedBn(pair.first, i);
        const BlobDesc& blob_desc = LogicalBlobDesc4BnInOp(bn);
        CHECK((&blob_desc) != nullptr);
        BlobDef blob_def(blob_desc.shape(), blob_desc.data_type());
        CHECK(bn2blob_def_.emplace(bn, blob_def).second);
      }
    }
  };
  InitBlobDefFromOpArgs(op.user_conf().input());
  InitBlobDefFromOpArgs(op.user_conf().output());
}

bool UserOpWrapper::NeedGenGradBlob4OpInput(const std::string& input_arg_name,
                                            int32_t index) const {
  auto it = op_conf().user_conf().input().find(input_arg_name);
  CHECK(it != op_conf().user_conf().input().end());
  CHECK(index >= 0 && index < it->second.s_size());
  return diff_fn_(GenRepeatedBn(input_arg_name, index)) != nullptr;
}

std::string UserOpWrapper::GetGradBlobWithOpOutput(const std::string& output_arg_name,
                                                   int32_t index) const {
  auto it = op_conf().user_conf().output().find(output_arg_name);
  CHECK(it != op_conf().user_conf().output().end());
  CHECK(index >= 0 && index < it->second.s_size());
  return GenLogicalBlobName(*diff_fn_(GenRepeatedBn(output_arg_name, index)));
}

void UserOpWrapper::BindGradBlobWithOpInput(const std::string logical_grad_blob_name,
                                            const std::string& input_arg_name,
                                            int32_t index) const {
  CHECK(NeedGenGradBlob4OpInput(input_arg_name, index));
  *diff_fn_(GenRepeatedBn(input_arg_name, index)) = GenLogicalBlobId(logical_grad_blob_name);
}

const BlobDef& UserOpWrapper::BlobDef4ArgNameAndIndex(const std::string& arg_name,
                                                      int32_t index) const {
  std::string bn = GenRepeatedBn(arg_name, index);
  CHECK(bn2blob_def_.find(bn) != bn2blob_def_.end());
  return bn2blob_def_.at(bn);
}

UserOpConfWrapperBuilder& UserOpConfWrapperBuilder::Input(const std::string& arg_name,
                                                          const std::string& logical_blob_name) {
  input_[arg_name].push_back(logical_blob_name);
  return *this;
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

UserOpConfWrapper UserOpConfWrapperBuilder::Build() {
  OperatorConf op_conf;
  op_conf.set_name(op_name_);
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
  wrapper_ = UserOpConfWrapper(op_conf);
  return wrapper_;
}

}  // namespace user_op

namespace {

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
        << UserOpAttrType_Name(attr.type());
  }
  return Maybe<void>::Ok();
}

Maybe<void> AddUserOpConfOutputDefaultArg(const UserOpDef& op_def, OperatorConf* op_conf) {
  UserOpConf* user_conf = op_conf->mutable_user_conf();
  // add default output arg and lbn
  for (const auto& output_arg : op_def.output()) {
    LOG(INFO) << "cclog: output arg_name: " << output_arg.name()
              << " is_optional: " << output_arg.is_optional()
              << " num_as_min: " << output_arg.num_as_min();
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

}  // namespace

Maybe<OperatorConf> CheckAndCompleteUserOpConfImpl(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_user_conf()) << " Add default value only for user op";
  OperatorConf ret = op_conf;
  UserOpConf* user_conf = ret.mutable_user_conf();
  const user_op::OpRegistrationVal* val = user_op::LookUpInOpRegistry(user_conf->op_type_name());
  CHECK_OR_RETURN(val) << " Cannot find op_type_name: " << user_conf->op_type_name();
  const UserOpDef& op_def = val->op_def;

  JUST(AddAttrDefaultValueAndCheckValid(op_def, &ret));
  JUST(AddUserOpConfOutputDefaultArg(op_def, &ret));
  // check input and output valid
  JUST(CheckArgDefIsValidInUserOpConf(op_conf, user_conf->input(), op_def.input()));
  JUST(CheckArgDefIsValidInUserOpConf(op_conf, user_conf->output(), op_def.output()));
  // check attr valid by user
  JUST(val->check_fn(user_op::UserOpDefWrapper(op_def), *user_conf));
  return ret;
}

}  // namespace oneflow
