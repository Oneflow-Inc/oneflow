#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

template<ConfigDefType config_def_type>
ConfigDef* MutGlobalConfigDef() {
  static ConfigDef config_def;
  return &config_def;
}

}  // namespace

const ConfigDef& GlobalEnvConfigDef() { return *MutGlobalConfigDef<kEnvConfigType>(); }
const ConfigDef& GlobalSessionConfigDef() { return *MutGlobalConfigDef<kSessionConfigType>(); }
const ConfigDef& GlobalFunctionConfigDef() { return *MutGlobalConfigDef<kFunctionConfigType>(); }

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Bool(
    const std::string& name, bool default_val) const {
  auto* flag2default = MutGlobalConfigDef<config_def_type>()->mutable_flag_name2default_val();
  CHECK(flag2default->find(name) == flag2default->end());
  (*flag2default)[name].set_at_bool(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Int64(
    const std::string& name, int64_t default_val) const {
  auto* flag2default = MutGlobalConfigDef<config_def_type>()->mutable_flag_name2default_val();
  CHECK(flag2default->find(name) == flag2default->end());
  (*flag2default)[name].set_at_int64(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Double(
    const std::string& name, double default_val) const {
  auto* flag2default = MutGlobalConfigDef<config_def_type>()->mutable_flag_name2default_val();
  CHECK(flag2default->find(name) == flag2default->end());
  (*flag2default)[name].set_at_double(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::String(
    const std::string& name, const std::string& default_val) const {
  auto* flag2default = MutGlobalConfigDef<config_def_type>()->mutable_flag_name2default_val();
  CHECK(flag2default->find(name) == flag2default->end());
  (*flag2default)[name].set_at_string(default_val);
  return *this;
}

template class ConfigDefBuidler<kEnvConfigType>;
template class ConfigDefBuidler<kSessionConfigType>;
template class ConfigDefBuidler<kFunctionConfigType>;

}  // namespace oneflow
