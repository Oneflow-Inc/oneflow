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
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

template<ConfigDefType config_def_type>
ConfigDef* MutGlobalConfigDef() {
  static ConfigDef config_def;
  return &config_def;
}

template<ConfigDefType config_def_type>
AttrValue* AddAttrDef(const std::string& name, const std::string& description) {
  auto* name2flag_def = MutGlobalConfigDef<config_def_type>()->mutable_attr_name2attr_def();
  CHECK(name2flag_def->find(name) == name2flag_def->end());
  auto* flag_def = &(*name2flag_def)[name];
  flag_def->set_name(name);
  flag_def->set_description(description);
  return flag_def->mutable_default_val();
}

}  // namespace

const ConfigDef& GlobalEnvConfigDef() { return *MutGlobalConfigDef<kEnvConfigDefType>(); }
const ConfigDef& GlobalSessionConfigDef() { return *MutGlobalConfigDef<kSessionConfigDefType>(); }
const ConfigDef& GlobalFunctionConfigDef() { return *MutGlobalConfigDef<kFunctionConfigDefType>(); }
const ConfigDef& GlobalScopeConfigDef() { return *MutGlobalConfigDef<kScopeConfigDefType>(); }

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Bool(
    const std::string& name, bool default_val, const std::string& description) const {
  AddAttrDef<config_def_type>(name, description)->set_at_bool(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Int64(
    const std::string& name, int64_t default_val, const std::string& description) const {
  AddAttrDef<config_def_type>(name, description)->set_at_int64(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Double(
    const std::string& name, double default_val, const std::string& description) const {
  AddAttrDef<config_def_type>(name, description)->set_at_double(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::String(
    const std::string& name, const std::string& default_val, const std::string& description) const {
  AddAttrDef<config_def_type>(name, description)->set_at_string(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::ListInt64(
    const std::string& name, const std::vector<int64_t>& default_val,
    const std::string& description) const {
  auto* list = AddAttrDef<config_def_type>(name, description)->mutable_at_list_int64();
  *list->mutable_val() = {default_val.begin(), default_val.end()};
  return *this;
}

template class ConfigDefBuidler<kEnvConfigDefType>;
template class ConfigDefBuidler<kSessionConfigDefType>;
template class ConfigDefBuidler<kFunctionConfigDefType>;
template class ConfigDefBuidler<kScopeConfigDefType>;

}  // namespace oneflow
