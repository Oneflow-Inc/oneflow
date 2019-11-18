#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

namespace {

HashMap<std::string, OpRegistrationVal>* MutOpRegistry() {
  static HashMap<std::string, OpRegistrationVal> registry;
  return &registry;
}

}  // namespace

void OpRegistryWrapper::InsertToGlobalRegistry() {
  auto registry = MutOpRegistry();
  registry->emplace(op_type_name, reg_val);
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
    *(wrapper_.reg_val.op_def.mutable_in()->Add()) = arg_def;
  } else {
    *(wrapper_.reg_val.op_def.mutable_out()->Add()) = arg_def;
  }
  return *this;
}

#define FN_PREFIX_SEQ                             \
  OF_PP_MAKE_TUPLE_SEQ(Input, true, false)        \
  OF_PP_MAKE_TUPLE_SEQ(OptionalInput, true, true) \
  OF_PP_MAKE_TUPLE_SEQ(Output, false, false)      \
  OF_PP_MAKE_TUPLE_SEQ(OptionalOutput, false, true)

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

OF_PP_FOR_EACH_TUPLE(OP_REG_ARG_MEMBER_FUNC, FN_PREFIX_SEQ)

#undef OP_REG_ARG_MEMBER_FUNC
#undef FN_PREFIX_SEQ

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr(const std::string& name,
                                                         UserOpAttrType type) {
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  *(wrapper_.reg_val.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetShapeInferFn(
    std::function<Maybe<void>(Shape4ArgNameAndIndex)> shape_infer_fn) {
  wrapper_.reg_val.shape_infer_fn = std::move(shape_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetDataTypeInferFn(
    std::function<Maybe<void>(Dtype4ArgNameAndIndex)> dtype_infer_fn) {
  wrapper_.reg_val.dtype_infer_fn = std::move(dtype_infer_fn);
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetGetSbpFn(
    std::function<Maybe<void>(/*TODO(niuchong): what is the para*/)>) {
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
