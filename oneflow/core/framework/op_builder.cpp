#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_attr.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/batch_axis_context.h"

namespace oneflow {

namespace user_op {

namespace {}  // namespace

// const OpRegistrationVal* LookUpInOpRegistry(const std::string& op_type_name) {
//   const auto registry = MutOpRegistry();
//   auto it = registry->find(op_type_name);
//   if (it != registry->end()) { return &(it->second); }
//   return nullptr;
// }
//
// std::vector<std::string> GetAllUserOpInOpRegistry() {
//   std::vector<std::string> ret;
//   const auto registry = MutOpRegistry();
//   for (auto it = registry->begin(); it != registry->end(); ++it) { ret.push_back(it->first); }
//   return ret;
// }

namespace {

bool InsertIfNotExists(const std::string& name, HashSet<std::string>* unique_names) {
  if (unique_names->find(name) != unique_names->end()) { return false; }
  unique_names->emplace(name);
  return true;
}

}  // namespace

OpBuilder& OpBuilder::Name(const std::string& op_type_name) {
  CHECK(InsertIfNotExists(op_type_name, &unique_names_));
  result_.op_type_name = op_type_name;
}

OpBuilder& OpBuilder::ArgImpl(bool is_input, const std::string& name, bool is_optional, int32_t num,
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
    *(result_.reg_val.op_def.mutable_input()->Add()) = arg_def;
  } else {
    *(result_.reg_val.op_def.mutable_output()->Add()) = arg_def;
  }
  return *this;
}

#define OP_REG_ARG_MEMBER_FUNC(name_prefix, is_input, is_optional)                           \
  OpBuilder& OpBuilder::name_prefix(const std::string& name) {                               \
    return ArgImpl(is_input, name, is_optional, 1, false);                                   \
  }                                                                                          \
  OpBuilder& OpBuilder::name_prefix(const std::string& name, int32_t num) {                  \
    return ArgImpl(is_input, name, is_optional, num, false);                                 \
  }                                                                                          \
  OpBuilder& OpBuilder::name_prefix##WithMinimum(const std::string& name, int32_t min_num) { \
    return ArgImpl(is_input, name, is_optional, min_num, true);                              \
  }

OP_REG_ARG_MEMBER_FUNC(Input, true, false)
OP_REG_ARG_MEMBER_FUNC(OptionalInput, true, true)
OP_REG_ARG_MEMBER_FUNC(Output, false, false)
OP_REG_ARG_MEMBER_FUNC(OptionalOutput, false, true)

#undef OP_REG_ARG_MEMBER_FUNC

OpBuilder& OpBuilder::SetOutputBufferNum(int32_t num) {
  result_.reg_val.same_output_regst_num = num;
  return *this;
}

OpBuilder& OpBuilder::SupportCpuOnly() {
  result_.reg_val.cpu_only_supported = true;
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& name, UserOpAttrType type) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  *(result_.reg_val.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

namespace {

void AddAttrWithDefault(OpBuildResult* result, const std::string& name, UserOpAttrType type,
                        std::function<void(UserOpDef::AttrDef*)> handler) {
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  handler(&attr_def);
  *(result->reg_val.op_def.mutable_attr()->Add()) = std::move(attr_def);
}

}  // namespace

#define ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                       \
  template<>                                                                               \
  OpBuilder& OpBuilder::Attr<cpp_type>(const std::string& name, UserOpAttrType type,       \
                                       cpp_type&& default_val) {                           \
    CHECK(InsertIfNotExists(name, &unique_names_));                                        \
    CHECK_EQ(type, attr_type);                                                             \
    AddAttrWithDefault(&result_, name, type, [default_val](UserOpDef::AttrDef* attr_def) { \
      AttrValAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());       \
    });                                                                                    \
    return *this;                                                                          \
  }

OF_PP_FOR_EACH_TUPLE(ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef ATTR_MEMBER_FUNC

OpBuilder& OpBuilder::SetTensorDescInferFn(TensorDescInferFn tensor_desc_infer_fn) {
  result_.reg_val.tensor_desc_infer_fn = std::move(tensor_desc_infer_fn);
  return *this;
}

OpBuilder& OpBuilder::SetBatchAxisInferFn(BatchAxisInferFn batch_axis_infer_fn) {
  result_.reg_val.batch_axis_infer_fn = std::move(batch_axis_infer_fn);
  return *this;
}

OpBuilder& OpBuilder::SetCheckAttrFn(CheckAttrFn fn) {
  result_.reg_val.check_fn = std::move(fn);
  return *this;
}

OpBuilder& OpBuilder::SetGetSbpFn(GetSbpFn get_sbp_fn) {
  result_.reg_val.get_sbp_fn = std::move(get_sbp_fn);
  return *this;
}

OpBuilder& OpBuilder::SetInputArgModifyFn(InputArgModifyFn input_arg_modify_fn) {
  result_.reg_val.input_arg_modify_fn = std::move(input_arg_modify_fn);
  return *this;
}

OpBuilder& OpBuilder::Finish() {
  CHECK(result_.reg_val.tensor_desc_infer_fn != nullptr)
      << "No TensorDescInfer function for " << result_.op_type_name;
  if (result_.reg_val.check_fn == nullptr) { result_.reg_val.check_fn = CheckAttrFnUtil::NoCheck; }
  if (result_.reg_val.batch_axis_infer_fn == nullptr) {
    result_.reg_val.batch_axis_infer_fn = BatchAxisInferFnUtil::DefaultAsFirstHasValueInput;
  }
  if (result_.reg_val.get_sbp_fn == nullptr) {
    result_.reg_val.get_sbp_fn = GetSbpFnUtil::DefaultBroadcastToBroadcast;
  }
  if (result_.reg_val.input_arg_modify_fn == nullptr) {
    result_.reg_val.input_arg_modify_fn = [](GetInputArgModifier, const UserOpConfWrapper&) {};
  }
  return *this;
}

OpBuildResult OpBuilder::GetResult() { return result_; }

}  // namespace user_op

}  // namespace oneflow
