#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_attr.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/batch_axis_context.h"

namespace oneflow {

namespace user_op {

namespace {

// only access with single thread
HashMap<std::string, OpRegistrationVal>* MutOpRegistry() {
  static HashMap<std::string, OpRegistrationVal> registry;
  return &registry;
}

}  // namespace

void OpRegistryWrapper::InsertToGlobalRegistry() {
  CHECK(!op_type_name.empty());
  auto registry = MutOpRegistry();
  CHECK(registry->emplace(op_type_name, reg_val).second);
}

const OpRegistrationVal* LookUpInOpRegistry(const std::string& op_type_name) {
  const auto registry = MutOpRegistry();
  auto it = registry->find(op_type_name);
  if (it != registry->end()) { return &(it->second); }
  return nullptr;
}

std::vector<std::string> GetAllUserOpInOpRegistry() {
  std::vector<std::string> ret;
  const auto registry = MutOpRegistry();
  for (auto it = registry->begin(); it != registry->end(); ++it) { ret.push_back(it->first); }
  return ret;
}

namespace {

bool InsertIfNotExists(const std::string& name, HashSet<std::string>* unique_names) {
  if (unique_names->find(name) != unique_names->end()) { return false; }
  unique_names->emplace(name);
  return true;
}

}  // namespace

OpRegistryWrapperBuilder::OpRegistryWrapperBuilder(const std::string& op_type_name) {
  CHECK(InsertIfNotExists(op_type_name, &unique_names_));
  wrapper_.op_type_name = op_type_name;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::ArgImpl(bool is_input, const std::string& name,
                                                            bool is_optional, int32_t num,
                                                            bool num_as_min) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::ArgDef arg_def;
  {
    arg_def.set_name(name);
    arg_def.set_is_optional(is_optional);
    arg_def.set_num(num);
    arg_def.set_num_as_min(num_as_min);
  }
  if (is_input) {
    *(wrapper_.reg_val.op_def.mutable_input()->Add()) = arg_def;
  } else {
    *(wrapper_.reg_val.op_def.mutable_output()->Add()) = arg_def;
  }
  return *this;
}

#define OP_REG_ARG_MEMBER_FUNC(name_prefix, is_input, is_optional)                           \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::name_prefix(const std::string& name) { \
    return ArgImpl(is_input, name, is_optional, 1, false);                                   \
  }                                                                                          \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::name_prefix(const std::string& name,   \
                                                                  int32_t num) {             \
    return ArgImpl(is_input, name, is_optional, num, false);                                 \
  }                                                                                          \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::name_prefix##WithMinimum(              \
      const std::string& name, int32_t min_num) {                                            \
    return ArgImpl(is_input, name, is_optional, min_num, true);                              \
  }

OP_REG_ARG_MEMBER_FUNC(Input, true, false)
OP_REG_ARG_MEMBER_FUNC(OptionalInput, true, true)
OP_REG_ARG_MEMBER_FUNC(Output, false, false)
OP_REG_ARG_MEMBER_FUNC(OptionalOutput, false, true)

#undef OP_REG_ARG_MEMBER_FUNC

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetOutputBufferNum(int32_t num) {
  wrapper_.reg_val.same_output_regst_num = num;
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SupportCpuOnly() {
  wrapper_.reg_val.cpu_only_supported = true;
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr(const std::string& name,
                                                         UserOpAttrType type) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  *(wrapper_.reg_val.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

namespace {

void AddAttrWithDefault(OpRegistryWrapper* wrapper, const std::string& name, UserOpAttrType type,
                        std::function<void(UserOpDef::AttrDef*)> handler) {
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  handler(&attr_def);
  *(wrapper->reg_val.op_def.mutable_attr()->Add()) = std::move(attr_def);
}

}  // namespace

#define ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                        \
  template<>                                                                                \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr<cpp_type>(                       \
      const std::string& name, UserOpAttrType type, cpp_type&& default_val) {               \
    CHECK(InsertIfNotExists(name, &unique_names_));                                         \
    CHECK_EQ(type, attr_type);                                                              \
    AddAttrWithDefault(&wrapper_, name, type, [default_val](UserOpDef::AttrDef* attr_def) { \
      AttrValAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());        \
    });                                                                                     \
    return *this;                                                                           \
  }

OF_PP_FOR_EACH_TUPLE(ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef ATTR_MEMBER_FUNC

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetTensorDescInferFn(
    TensorDescInferFn tensor_desc_infer_fn) {
  wrapper_.reg_val.tensor_desc_infer_fn = std::move(tensor_desc_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetBatchAxisInferFn(
    BatchAxisInferFn batch_axis_infer_fn) {
  wrapper_.reg_val.batch_axis_infer_fn = std::move(batch_axis_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetCheckAttrFn(CheckAttrFn fn) {
  wrapper_.reg_val.check_fn = std::move(fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetGetSbpFn(GetSbpFn get_sbp_fn) {
  wrapper_.reg_val.get_sbp_fn = std::move(get_sbp_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetInputArgModifyFn(
    InputArgModifyFn input_arg_modify_fn) {
  wrapper_.reg_val.input_arg_modify_fn = std::move(input_arg_modify_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetOutputArgModifyFn(
    OutputArgModifyFn output_arg_modify_fn) {
  wrapper_.reg_val.output_arg_modify_fn = std::move(output_arg_modify_fn);
  return *this;
}

OpRegistryWrapper OpRegistryWrapperBuilder::Build() {
  CHECK(wrapper_.reg_val.tensor_desc_infer_fn != nullptr)
      << "No TensorDescInfer function for " << wrapper_.op_type_name;
  if (wrapper_.reg_val.check_fn == nullptr) {
    wrapper_.reg_val.check_fn = CheckAttrFnUtil::NoCheck;
  }
  if (wrapper_.reg_val.batch_axis_infer_fn == nullptr) {
    wrapper_.reg_val.batch_axis_infer_fn = BatchAxisInferFnUtil::DefaultAsFirstHasValueInput;
  }
  if (wrapper_.reg_val.get_sbp_fn == nullptr) {
    wrapper_.reg_val.get_sbp_fn = GetSbpFnUtil::DefaultBroadcastToBroadcast;
  }
  if (wrapper_.reg_val.input_arg_modify_fn == nullptr) {
    wrapper_.reg_val.input_arg_modify_fn = [](GetInputArgModifier, const UserOpConfWrapper&) {};
  }
  if (wrapper_.reg_val.output_arg_modify_fn == nullptr) {
    wrapper_.reg_val.output_arg_modify_fn = [](GetOutputArgModifier, const UserOpConfWrapper&) {};
  }
  return wrapper_;
}

}  // namespace user_op

}  // namespace oneflow
