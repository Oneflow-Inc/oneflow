#ifndef ONEFLOW_CORE_JOB_CONFIG_DEF_H_
#define ONEFLOW_CORE_JOB_CONFIG_DEF_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/config_def.pb.h"

namespace oneflow {

template<ConfigDefType config_def_type>
struct ConfigDefBuidler final {
  const ConfigDefBuidler& Bool(const std::string& name, bool default_val,
                               const std::string& description) const;
  const ConfigDefBuidler& Int64(const std::string& name, int64_t default_val,
                                const std::string& description) const;
  const ConfigDefBuidler& Double(const std::string& name, double default_val,
                                 const std::string& description) const;
  const ConfigDefBuidler& String(const std::string& name, const std::string& default_val,
                                 const std::string& description) const;
};

#define REGISTER_ENV_CONFIG_DEF() REGISTER_CONFIG_DEF(kEnvConfigType)
#define REGISTER_SESSION_CONFIG_DEF() REGISTER_CONFIG_DEF(kSessionConfigType)
#define REGISTER_FUNCTION_CONFIG_DEF() REGISTER_CONFIG_DEF(kFunctionConfigType)

#define REGISTER_CONFIG_DEF(config_def_type)                                                 \
  static ConfigDefBuidler<config_def_type> OF_PP_CAT(g_##config_def_type##_def_, __LINE__) = \
      ConfigDefBuidler<config_def_type>()

const ConfigDef& GlobalEnvConfigDef();
const ConfigDef& GlobalSessionConfigDef();
const ConfigDef& GlobalFunctionConfigDef();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CONFIG_DEF_H_
