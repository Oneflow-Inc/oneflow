#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/user_op_attr.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

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

OpRegistryWrapperBuilder::OpRegistryWrapperBuilder(const std::string& op_type_name) {
  wrapper_.op_type_name = op_type_name;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::ArgImpl(bool is_input, const std::string& name,
                                                            bool is_optional, int32_t num,
                                                            bool num_as_min) {
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

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr(const std::string& name,
                                                         UserOpAttrType type) {
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

#define OP_REG_ATTR_MEMBER_FUNC(attr_type, cpp_type, postfix)                               \
  template<>                                                                                \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr<cpp_type>(                       \
      const std::string& name, UserOpAttrType type, cpp_type&& default_val) {               \
    CHECK_EQ(type, UserOpAttrType::kAt##attr_type);                                         \
    AddAttrWithDefault(&wrapper_, name, type, [default_val](UserOpDef::AttrDef* attr_def) { \
      attr_def->mutable_default_val()->set_##postfix(default_val);                          \
    });                                                                                     \
    return *this;                                                                           \
  }

OP_REG_ATTR_MEMBER_FUNC(Int32, int32_t, at_int32)
OP_REG_ATTR_MEMBER_FUNC(Int64, int64_t, at_int64)
OP_REG_ATTR_MEMBER_FUNC(Bool, bool, at_bool)
OP_REG_ATTR_MEMBER_FUNC(Float, float, at_float)
OP_REG_ATTR_MEMBER_FUNC(Double, double, at_double)
OP_REG_ATTR_MEMBER_FUNC(String, std::string, at_string)

#undef OP_REG_ATTR_MEMBER_FUNC

template<>
OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr<Shape>(const std::string& name,
                                                                UserOpAttrType type,
                                                                Shape&& default_val) {
  CHECK_EQ(type, UserOpAttrType::kAtShape);
  AddAttrWithDefault(&wrapper_, name, type, [default_val](UserOpDef::AttrDef* attr_def) {
    default_val.ToProto(attr_def->mutable_default_val()->mutable_at_shape());
  });
  return *this;
}

#define OP_REG_LIST_ATTR_MEMBER_FUNC(attr_type, cpp_type, postfix)                          \
  template<>                                                                                \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr<cpp_type>(                       \
      const std::string& name, UserOpAttrType type, cpp_type&& default_val) {               \
    CHECK_EQ(type, UserOpAttrType::kAt##attr_type);                                         \
    AddAttrWithDefault(&wrapper_, name, type, [default_val](UserOpDef::AttrDef* attr_def) { \
      SerializeVector2ListAttr<cpp_type, UserOpAttrVal::attr_type>(                         \
          default_val, attr_def->mutable_default_val()->mutable_##postfix());               \
    });                                                                                     \
    return *this;                                                                           \
  }

OP_REG_LIST_ATTR_MEMBER_FUNC(ListInt32, std::vector<int32_t>, at_list_int32)
OP_REG_LIST_ATTR_MEMBER_FUNC(ListInt64, std::vector<int64_t>, at_list_int64)
OP_REG_LIST_ATTR_MEMBER_FUNC(ListFloat, std::vector<float>, at_list_float)

#undef OP_REG_LIST_ATTR_MEMBER_FUNC

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetShapeInferFn(ShapeInferFn shape_infer_fn) {
  wrapper_.reg_val.shape_infer_fn = std::move(shape_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetDataTypeInferFn(
    DtypeInferFn dtype_infer_fn) {
  wrapper_.reg_val.dtype_infer_fn = std::move(dtype_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetGetSbpFn(GetSbpFn get_sbp_fn) {
  wrapper_.reg_val.get_sbp_fn = std::move(get_sbp_fn);
  return *this;
}

OpRegistryWrapper OpRegistryWrapperBuilder::Build() {
  if (wrapper_.reg_val.check_fn == nullptr) {
    wrapper_.reg_val.check_fn = [](const UserOpDef&, const UserOpConf&) {
      return Maybe<void>::Ok();
    };
  }
  if (wrapper_.reg_val.shape_infer_fn == nullptr) {
    wrapper_.reg_val.shape_infer_fn = [](Shape4ArgNameAndIndex) {
      // TODO(niuchong): default impl
      return Maybe<void>::Ok();
    };
  }
  if (wrapper_.reg_val.dtype_infer_fn == nullptr) {
    wrapper_.reg_val.dtype_infer_fn = [](Dtype4ArgNameAndIndex) {
      // TODO(niuchong): default impl
      return Maybe<void>::Ok();
    };
  }
  if (wrapper_.reg_val.get_sbp_fn == nullptr) {
    wrapper_.reg_val.get_sbp_fn = [](/**/) {
      // TODO(niuchong): default impl
      return Maybe<void>::Ok();
    };
  }
  return wrapper_;
}

}  // namespace user_op

}  // namespace oneflow
