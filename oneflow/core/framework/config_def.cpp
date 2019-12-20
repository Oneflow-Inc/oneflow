#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

template<ConfigDefType config_def_type>
ConfigDef* MutGlobalConfigDef() {
  static ConfigDef config_def;
  return &config_def;
}

template<ConfigDefType config_def_type>
UserOpAttrVal* AddConfigFlagDef(const std::string& name, const std::string& description) {
  auto* name2flag_def = MutGlobalConfigDef<config_def_type>()->mutable_flag_name2flag_def();
  CHECK(name2flag_def->find(name) == name2flag_def->end());
  auto* flag_def = &(*name2flag_def)[name];
  flag_def->set_name(name);
  flag_def->set_description(description);
  return flag_def->mutable_default_val();
}

}  // namespace

const ConfigDef& GlobalEnvConfigDef() { return *MutGlobalConfigDef<kEnvConfigType>(); }
const ConfigDef& GlobalSessionConfigDef() { return *MutGlobalConfigDef<kSessionConfigType>(); }
const ConfigDef& GlobalFunctionConfigDef() { return *MutGlobalConfigDef<kFunctionConfigType>(); }

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Bool(
    const std::string& name, bool default_val, const std::string& description) const {
  AddConfigFlagDef<config_def_type>(name, description)->set_at_bool(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Int64(
    const std::string& name, int64_t default_val, const std::string& description) const {
  AddConfigFlagDef<config_def_type>(name, description)->set_at_int64(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Double(
    const std::string& name, double default_val, const std::string& description) const {
  AddConfigFlagDef<config_def_type>(name, description)->set_at_double(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::String(
    const std::string& name, const std::string& default_val, const std::string& description) const {
  AddConfigFlagDef<config_def_type>(name, description)->set_at_string(default_val);
  return *this;
}

template class ConfigDefBuidler<kEnvConfigType>;
template class ConfigDefBuidler<kSessionConfigType>;
template class ConfigDefBuidler<kFunctionConfigType>;

}  // namespace oneflow
