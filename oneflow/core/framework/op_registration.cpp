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
}

#define FN_PREFIX_SEQ                             \
  OF_PP_MAKE_TUPLE_SEQ(Input, true, false)        \
  OF_PP_MAKE_TUPLE_SEQ(OptionalInput, true, true) \
  OF_PP_MAKE_TUPLE_SEQ(Output, false, false)      \
  OF_PP_MAKE_TUPLE_SEQ(OptionalOutput, false, true)

#define OP_REG_ARG_MEMBER_FUNC(fn_prefix_tuple)                                                   \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::##OF_PP_TUPLE_ELEM(                         \
      0, fn_prefix_tuple)##(const std::string& name) {                                            \
    return ArgImpl(OF_PP_TUPLE_ELEM(1, fn_prefix), name, OF_PP_TUPLE_ELEM(2, fn_prefix), 1,       \
                   false);                                                                        \
  }                                                                                               \
  OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::##OF_PP_TUPLE_ELEM(0, fn_prefix_tuple)##(   \
      const std::string& name, int32_t num) {                                                     \
    return ArgImpl(OF_PP_TUPLE_ELEM(1, fn_prefix), name, OF_PP_TUPLE_ELEM(2, fn_prefix), num,     \
                   false);                                                                        \
  }                                                                                               \
  OpRegistryWrapperBuilderder& OpRegistryWrapperBuilder::##OF_PP_TUPLE_ELEM(                      \
      0, fn_prefix_tuple)##WithMinimum(const std::string& name, int32_t min_num) {                \
    return ArgImpl(OF_PP_TUPLE_ELEM(1, fn_prefix), name, OF_PP_TUPLE_ELEM(2, fn_prefix), min_num, \
                   true);                                                                         \
  }

/*
OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Input(const std::string& name) {
  return ArgImpl(true, name, false, 1, false);
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Input(const std::string& name, int32_t num) {
  return ArgImpl(true, name, false, num, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::InputWithMinimum(const std::string& name,
int32_t min_num) { return ArgImpl(true, name, false, min_num, true);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalInput(const std::string& name) {
  return ArgImpl(true, name, true, 1, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalInput(const std::string& name,
int32_t num) { return ArgImpl(true, name, true, num, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalInputWithMinimum(const std::string&
name, int32_t min_num) { return ArgImpl(true, name, true, min_num, true);
}

OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::Output(const std::string& name) {
  return ArgImpl(false, name, false, 1, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::Output(const std::string& name, int32_t num)
{ return ArgImpl(false, name, false, num, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OutputWithMinimum(const std::string& name,
int32_t min_num) { return ArgImpl(false, name, false, min_num, true);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalOutput(const std::string& name) {
  return ArgImpl(false, name, true, 1, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalOutput(const std::string& name,
int32_t num) { return ArgImpl(false, name, true, num, false);
}
OpRegistryWrapperBuilderder&  OpRegistryWrapperBuilder::OptionalOutputWithMinimum(const std::string&
name, int32_t min_num) { return ArgImpl(false, name, true, min_num, true);
}
*/

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::Attr(const std::string& name,
                                                         UserOpAttrType type) {
  UserOpDef::AttrDef attr_def;
  attr_def.name = name;
  attr_def.type = type;
  *(wrapper_.reg_val.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetShapeInferFn(
    std::function<Maybe<void>(Shape4ArgNameAndIndex)> shape_infer_fn) {
  wrapper_.reg_val.shape_infer_fn = std::move(shape_infer_fn);
}

OpRegistryWrapperBuilder& OpRegistryWrapperBuilder::SetDataTypeInferFn(
    std::function<Maybe<void>(Dtype4ArgNameAndIndex)> dtype_infer_fn) {
  wrapper_.reg_val.dtype_infer_fn = std::move(dtype_infer_fn);
}

}  // namespace user_op

}  // namespace oneflow
