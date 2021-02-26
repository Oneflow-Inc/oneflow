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
#include "oneflow/core/framework/user_op_registry.h"

#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/sbp_context.h"

namespace oneflow {

namespace user_op {

namespace {

bool InsertIfNotExists(const std::string& name, HashSet<std::string>* unique_names) {
  if (unique_names->find(name) != unique_names->end()) { return false; }
  unique_names->emplace(name);
  return true;
}

}  // namespace

OpRegistry& OpRegistry::Name(const std::string& op_type_name) {
  CHECK(InsertIfNotExists(op_type_name, &unique_names_));
  result_.op_type_name = op_type_name;
  return *this;
}

OpRegistry& OpRegistry::ArgImpl(bool is_input, const std::string& name, bool is_optional,
                                int32_t num, bool num_as_min) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::ArgDef arg_def;
  {
    arg_def.set_name(name);
    arg_def.set_is_optional(is_optional);
    arg_def.set_num(num);
    arg_def.set_num_as_min(num_as_min);
  }
  if (is_input) {
    *(result_.op_def.mutable_input()->Add()) = arg_def;
  } else {
    *(result_.op_def.mutable_output()->Add()) = arg_def;
  }
  return *this;
}

#define OP_REG_ARG_MEMBER_FUNC(name_prefix, is_input, is_optional)                             \
  OpRegistry& OpRegistry::name_prefix(const std::string& name) {                               \
    return ArgImpl(is_input, name, is_optional, 1, false);                                     \
  }                                                                                            \
  OpRegistry& OpRegistry::name_prefix(const std::string& name, int32_t num) {                  \
    return ArgImpl(is_input, name, is_optional, num, false);                                   \
  }                                                                                            \
  OpRegistry& OpRegistry::name_prefix##WithMinimum(const std::string& name, int32_t min_num) { \
    return ArgImpl(is_input, name, is_optional, min_num, true);                                \
  }

OP_REG_ARG_MEMBER_FUNC(Input, true, false)
OP_REG_ARG_MEMBER_FUNC(OptionalInput, true, true)
OP_REG_ARG_MEMBER_FUNC(Output, false, false)
OP_REG_ARG_MEMBER_FUNC(OptionalOutput, false, true)

#undef OP_REG_ARG_MEMBER_FUNC

OpRegistry& OpRegistry::SupportCpuOnly() {
  result_.cpu_only_supported = true;
  return *this;
}

OpRegistry& OpRegistry::SetOutputBufferNum(int32_t num) {
  result_.same_output_regst_num = num;
  return *this;
}

OpRegistry& OpRegistry::Attr(const std::string& name, AttrType type) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  *(result_.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

namespace {

void AddAttrWithDefault(OpRegistryResult* result, const std::string& name, AttrType type,
                        std::function<void(UserOpDef::AttrDef*)> handler) {
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  handler(&attr_def);
  *(result->op_def.mutable_attr()->Add()) = std::move(attr_def);
}

}  // namespace

#define ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                             \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name, AttrType type,                 \
                                         const cpp_type& default_val) {                          \
    CHECK_EQ(type, attr_type);                                                                   \
    return DefaultedAttr(name, type, [default_val](UserOpDef::AttrDef* attr_def) {               \
      AttrValueAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());           \
    });                                                                                          \
  }                                                                                              \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name, const cpp_type& default_val) { \
    return DefaultedAttr(                                                                        \
        name, GetAttrType<cpp_type>::value, [default_val](UserOpDef::AttrDef* attr_def) {        \
          AttrValueAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());       \
        });                                                                                      \
  }                                                                                              \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name) {                              \
    return Attr<cpp_type>(name, cpp_type());                                                     \
  }

OF_PP_FOR_EACH_TUPLE(ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef ATTR_MEMBER_FUNC

OpRegistry& OpRegistry::DefaultedAttr(const std::string& name, AttrType type,
                                      const std::function<void(UserOpDef::AttrDef*)>& SetDefault) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  AddAttrWithDefault(&result_, name, type, SetDefault);
  return *this;
}

OpRegistry& OpRegistry::SetTensorDescInferFn(TensorDescInferFn tensor_desc_infer_fn) {
  result_.tensor_desc_infer_fn = std::move(tensor_desc_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetCheckAttrFn(CheckAttrFn fn) {
  result_.check_fn = std::move(fn);
  return *this;
}

OpRegistry& OpRegistry::SetGetSbpFn(GetSbpFn get_sbp_fn) {
  result_.get_sbp_fn = std::move(get_sbp_fn);
  return *this;
}

OpRegistry& OpRegistry::SetInferSbpSignatureFn(InferSbpSignatureFn infer_sbp_signature_fn) {
  result_.infer_sbp_signature_fn = std::move(infer_sbp_signature_fn);
  return *this;
}

OpRegistry& OpRegistry::SetInputArgModifyFn(InputArgModifyFn input_arg_modify_fn) {
  result_.input_arg_modify_fn = std::move(input_arg_modify_fn);
  return *this;
}

OpRegistry& OpRegistry::SetOutputArgModifyFn(OutputArgModifyFn output_arg_modify_fn) {
  result_.output_arg_modify_fn = std::move(output_arg_modify_fn);
  return *this;
}

OpRegistry& OpRegistry::SetInferOutputBlobTimeShapeFn(
    InferOutputBlobTimeShapeFn infer_output_blob_time_shape_fn) {
  result_.infer_output_blob_time_shape_fn = std::move(infer_output_blob_time_shape_fn);
  return *this;
}

OpRegistry& OpRegistry::Finish() {
  CHECK(result_.tensor_desc_infer_fn != nullptr)
      << "No TensorDescInfer function for " << result_.op_type_name;
  if (result_.check_fn == nullptr) { result_.check_fn = CheckAttrFnUtil::NoCheck; }
  if (result_.get_sbp_fn == nullptr) {
    result_.get_sbp_fn = GetSbpFnUtil::DefaultBroadcastToBroadcast;
  }
  if (result_.input_arg_modify_fn == nullptr) {
    result_.input_arg_modify_fn = [](GetInputArgModifier, const UserOpConfWrapper&) {};
  }
  if (result_.output_arg_modify_fn == nullptr) {
    result_.output_arg_modify_fn = [](GetOutputArgModifier, const UserOpConfWrapper&) {};
  }
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
