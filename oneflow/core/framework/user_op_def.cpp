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
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

namespace user_op {

UserOpDefWrapper::UserOpDefWrapper(const UserOpDef& def)
    : def_(def), inputs_(), outputs_(), attrs_() {
  for (int32_t i = 0; i < def_.input_size(); ++i) {
    inputs_.emplace(def_.input(i).name(), def_.mutable_input(i));
  }
  for (int32_t i = 0; i < def_.output_size(); ++i) {
    outputs_.emplace(def_.output(i).name(), def_.mutable_output(i));
  }
  for (int32_t i = 0; i < def_.attr_size(); ++i) {
    attrs_.emplace(def_.attr(i).name(), def_.mutable_attr(i));
  }
}

bool UserOpDefWrapper::IsInputArgName(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

bool UserOpDefWrapper::IsOutputArgName(const std::string& name) const {
  return outputs_.find(name) != outputs_.end();
}

bool UserOpDefWrapper::IsAttrName(const std::string& name) const {
  return attrs_.find(name) != attrs_.end();
}

bool UserOpDefWrapper::IsArgOptional(const std::string& name) const {
  const UserOpDef::ArgDef* arg_def = GetArgPointer(name);
  CHECK_NOTNULL(arg_def);
  return arg_def->is_optional();
}

std::pair<int32_t, bool> UserOpDefWrapper::ArgNumAndIsMin(const std::string& name) const {
  const UserOpDef::ArgDef* arg_def = GetArgPointer(name);
  CHECK_NOTNULL(arg_def);
  return std::make_pair(arg_def->num(), arg_def->num_as_min());
}

const UserOpDef::ArgDef* UserOpDefWrapper::GetArgPointer(const std::string& name) const {
  auto it = inputs_.find(name);
  if (it != inputs_.end()) { return it->second; }
  it = outputs_.find(name);
  if (it != outputs_.end()) { return it->second; }
  return nullptr;
}

AttrType UserOpDefWrapper::GetAttrType(const std::string& name) const {
  return attrs_.at(name)->type();
}

bool UserOpDefWrapper::AttrHasDefaultVal(const std::string& name) const {
  return attrs_.at(name)->has_default_val();
}

#define ATTR_TYPE_SPECIALIZATION(field, cpp_type, attr_type)                              \
  template<>                                                                              \
  cpp_type UserOpDefWrapper::GetAttrDefaultVal<cpp_type>(const std::string& name) const { \
    CHECK(AttrHasDefaultVal(name));                                                       \
    const AttrValue& default_val = attrs_.at(name)->default_val();                        \
    CHECK_EQ(static_cast<int>(attr_type), default_val.value_case());                      \
    return AttrValueAccessor<cpp_type>::Attr(default_val);                                \
  }

OF_PP_FOR_EACH_TUPLE(ATTR_TYPE_SPECIALIZATION, ATTR_SEQ)

#undef ATTR_TYPE_SPECIALIZATION
}  // namespace user_op

}  // namespace oneflow
