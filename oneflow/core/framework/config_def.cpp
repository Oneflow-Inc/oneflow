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
void CheckNoExistedField(const std::string& name) {
  const auto& flags = MutGlobalConfigDef<config_def_type>()->flag();
  auto Found = [&](const ConfigFlagDef& existed) { return existed.name() == name; };
  CHECK(std::find_if(flags.begin(), flags.end(), Found) == flags.end());
}

}  // namespace

const ConfigDef& GlobalEnvConfigDef() { return *MutGlobalConfigDef<kEnvConfigType>(); }
const ConfigDef& GlobalSessionConfigDef() { return *MutGlobalConfigDef<kSessionConfigType>(); }
const ConfigDef& GlobalFunctionConfigDef() { return *MutGlobalConfigDef<kFunctionConfigType>(); }

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Bool(
    const std::string& name, bool default_val) const {
  CheckNoExistedField<config_def_type>(name);
  auto* flag = MutGlobalConfigDef<config_def_type>()->mutable_flag()->Add();
  flag->set_name(name);
  flag->set_type(UserOpAttrType::kAtBool);
  flag->mutable_default_val()->set_at_bool(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Int64(
    const std::string& name, int64_t default_val) const {
  CheckNoExistedField<config_def_type>(name);
  auto* flag = MutGlobalConfigDef<config_def_type>()->mutable_flag()->Add();
  flag->set_name(name);
  flag->set_type(UserOpAttrType::kAtInt64);
  flag->mutable_default_val()->set_at_int64(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::Double(
    const std::string& name, double default_val) const {
  CheckNoExistedField<config_def_type>(name);
  auto* flag = MutGlobalConfigDef<config_def_type>()->mutable_flag()->Add();
  flag->set_name(name);
  flag->set_type(UserOpAttrType::kAtDouble);
  flag->mutable_default_val()->set_at_double(default_val);
  return *this;
}

template<ConfigDefType config_def_type>
const ConfigDefBuidler<config_def_type>& ConfigDefBuidler<config_def_type>::String(
    const std::string& name, const std::string& default_val) const {
  CheckNoExistedField<config_def_type>(name);
  auto* flag = MutGlobalConfigDef<config_def_type>()->mutable_flag()->Add();
  flag->set_name(name);
  flag->set_type(UserOpAttrType::kAtString);
  flag->mutable_default_val()->set_at_string(default_val);
  return *this;
}

template class ConfigDefBuidler<kEnvConfigType>;
template class ConfigDefBuidler<kSessionConfigType>;
template class ConfigDefBuidler<kFunctionConfigType>;

}  // namespace oneflow
