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
#include "oneflow/core/framework/attr.h"

namespace oneflow {

bool AttrsAccessor::HasAttrValue(const std::string& attr_name) const {
  return attrs().attr_name2attr_value().find(attr_name) != attrs().attr_name2attr_value().end();
}

Maybe<const AttrValue&> AttrsAccessor::GetAttrValue(const std::string& attr_name) const {
  const auto& iter = attrs().attr_name2attr_value().find(attr_name);
  CHECK_OR_RETURN(iter != attrs().attr_name2attr_value().end());
  return iter->second;
}

Maybe<bool> AttrsAccessor::Bool(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrValue(attr_name));
  CHECK_OR_RETURN(attr_value.has_at_bool());
  return attr_value.at_bool();
}

Maybe<int64_t> AttrsAccessor::Int64(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrValue(attr_name));
  CHECK_OR_RETURN(attr_value.has_at_int64());
  return attr_value.at_int64();
}

Maybe<double> AttrsAccessor::Double(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrValue(attr_name));
  CHECK_OR_RETURN(attr_value.has_at_double());
  return attr_value.at_double();
}

Maybe<const std::string&> AttrsAccessor::String(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrValue(attr_name));
  CHECK_OR_RETURN(attr_value.has_at_string());
  return attr_value.at_string();
}

Maybe<const PbRf<int64_t>&> AttrsAccessor::ListInt64(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrValue(attr_name));
  CHECK_OR_RETURN(attr_value.has_at_list_int64());
  return attr_value.at_list_int64().val();
}

AttrValue* AttrsMutAccessor::MutAttrValue(const std::string& attr_name) const {
  return &(*mut_attrs()->mutable_attr_name2attr_value())[attr_name];
}

const AttrsMutAccessor& AttrsMutAccessor::Bool(const std::string& attr_name,
                                               bool default_val) const {
  MutAttrValue(attr_name)->set_at_bool(default_val);
  return *this;
}

const AttrsMutAccessor& AttrsMutAccessor::Int64(const std::string& attr_name,
                                                int64_t default_val) const {
  MutAttrValue(attr_name)->set_at_int64(default_val);
  return *this;
}

const AttrsMutAccessor& AttrsMutAccessor::Double(const std::string& attr_name,
                                                 double default_val) const {
  MutAttrValue(attr_name)->set_at_double(default_val);
  return *this;
}
const AttrsMutAccessor& AttrsMutAccessor::String(const std::string& attr_name,
                                                 const std::string& default_val) const {
  MutAttrValue(attr_name)->set_at_string(default_val);
  return *this;
}

const AttrsMutAccessor& AttrsMutAccessor::ListInt64(const std::string& attr_name,
                                                    const std::vector<int64_t>& default_val) const {
  auto* list = MutAttrValue(attr_name)->mutable_at_list_int64();
  *list->mutable_val() = {default_val.begin(), default_val.end()};
  return *this;
}

bool AttrDefsAccessor::HasAttrDef(const std::string& attr_name) const {
  const auto& iter = def().attr_name2attr_index().find(attr_name);
  return iter != def().attr_name2attr_index().end();
}

Maybe<const AttrDef&> AttrDefsAccessor::GetAttrDef(const std::string& attr_name) const {
  const auto& iter = def().attr_name2attr_index().find(attr_name);
  CHECK_OR_RETURN(iter != def().attr_name2attr_index().end());
  CHECK_GE_OR_RETURN(iter->second, 0);
  CHECK_LT_OR_RETURN(iter->second, def().attr_def_size());
  return def().attr_def(iter->second);
}

Maybe<bool> AttrDefsAccessor::Bool(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrDef(attr_name)).default_val();
  CHECK_OR_RETURN(attr_value.has_at_bool());
  CHECK_EQ_OR_RETURN(static_cast<int>(attr_value.value_case()), static_cast<int>(kAtBool));
  return attr_value.at_bool();
}

Maybe<int64_t> AttrDefsAccessor::Int64(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrDef(attr_name)).default_val();
  CHECK_OR_RETURN(attr_value.has_at_int64());
  CHECK_EQ_OR_RETURN(static_cast<int>(attr_value.value_case()), static_cast<int>(kAtInt64));
  return attr_value.at_int64();
}

Maybe<double> AttrDefsAccessor::Double(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrDef(attr_name)).default_val();
  CHECK_OR_RETURN(attr_value.has_at_double());
  CHECK_EQ_OR_RETURN(static_cast<int>(attr_value.value_case()), static_cast<int>(kAtDouble));
  return attr_value.at_double();
}

Maybe<const std::string&> AttrDefsAccessor::String(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrDef(attr_name)).default_val();
  CHECK_OR_RETURN(attr_value.has_at_string());
  CHECK_EQ_OR_RETURN(static_cast<int>(attr_value.value_case()), static_cast<int>(kAtString));
  return attr_value.at_string();
}

Maybe<const PbRf<int64_t>&> AttrDefsAccessor::ListInt64(const std::string& attr_name) const {
  const auto& attr_value = JUST(GetAttrDef(attr_name)).default_val();
  CHECK_OR_RETURN(attr_value.has_at_list_int64());
  CHECK_EQ_OR_RETURN(static_cast<int>(attr_value.value_case()), static_cast<int>(kAtListInt64));
  return attr_value.at_list_int64().val();
}

AttrDef* AttrDefsMutAccessor::AddAttrDef(const std::string& attr_name) const {
  CHECK(def().attr_name2attr_index().find(attr_name) == def().attr_name2attr_index().end());
  (*mut_def()->mutable_attr_name2attr_index())[attr_name] = def().attr_def_size();
  auto* attr_def = mut_def()->mutable_attr_def()->Add();
  attr_def->set_name(attr_name);
  return attr_def;
}

const AttrDefsMutAccessor& AttrDefsMutAccessor::Bool(const std::string& attr_name, bool default_val,
                                                     const std::string& description) const {
  auto* attr_def = AddAttrDef(attr_name);
  attr_def->set_description(description);
  attr_def->set_type(kAtBool);
  attr_def->mutable_default_val()->set_at_bool(default_val);
  return *this;
}

const AttrDefsMutAccessor& AttrDefsMutAccessor::Int64(const std::string& attr_name,
                                                      int64_t default_val,
                                                      const std::string& description) const {
  auto* attr_def = AddAttrDef(attr_name);
  attr_def->set_description(description);
  attr_def->set_type(kAtInt64);
  attr_def->mutable_default_val()->set_at_int64(default_val);
  return *this;
}

const AttrDefsMutAccessor& AttrDefsMutAccessor::Double(const std::string& attr_name,
                                                       double default_val,
                                                       const std::string& description) const {
  auto* attr_def = AddAttrDef(attr_name);
  attr_def->set_description(description);
  attr_def->set_type(kAtDouble);
  attr_def->mutable_default_val()->set_at_double(default_val);
  return *this;
}
const AttrDefsMutAccessor& AttrDefsMutAccessor::String(const std::string& attr_name,
                                                       const std::string& default_val,
                                                       const std::string& description) const {
  auto* attr_def = AddAttrDef(attr_name);
  attr_def->set_description(description);
  attr_def->set_type(kAtString);
  attr_def->mutable_default_val()->set_at_string(default_val);
  return *this;
}

const AttrDefsMutAccessor& AttrDefsMutAccessor::ListInt64(const std::string& attr_name,
                                                          const std::vector<int64_t>& default_val,
                                                          const std::string& description) const {
  auto* attr_def = AddAttrDef(attr_name);
  attr_def->set_description(description);
  attr_def->set_type(kAtListInt64);
  auto* list = attr_def->mutable_default_val()->mutable_at_list_int64();
  *list->mutable_val() = {default_val.begin(), default_val.end()};
  return *this;
}

bool DefaultedAttrsAccessor::HasAttrValue(const std::string& attr_name) const {
  return attrs_accessor().HasAttrValue(attr_name) || defs_accessor().HasAttrDef(attr_name);
}

Maybe<const AttrValue&> DefaultedAttrsAccessor::GetAttrValue(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().GetAttrValue(attr_name); }
  return JUST(defs_accessor().GetAttrDef(attr_name)).default_val();
}

// Get values
Maybe<bool> DefaultedAttrsAccessor::Bool(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().Bool(attr_name); }
  return defs_accessor().Bool(attr_name);
}

Maybe<int64_t> DefaultedAttrsAccessor::Int64(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().Int64(attr_name); }
  return defs_accessor().Int64(attr_name);
}

Maybe<double> DefaultedAttrsAccessor::Double(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().Double(attr_name); }
  return defs_accessor().Double(attr_name);
}

Maybe<const std::string&> DefaultedAttrsAccessor::String(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().String(attr_name); }
  return defs_accessor().String(attr_name);
}

Maybe<const PbRf<int64_t>&> DefaultedAttrsAccessor::ListInt64(const std::string& attr_name) const {
  if (attrs_accessor().HasAttrValue(attr_name)) { return attrs_accessor().ListInt64(attr_name); }
  return defs_accessor().ListInt64(attr_name);
}

}  // namespace oneflow
