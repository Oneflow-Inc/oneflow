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

  const ConfigDefBuidler& ListInt64(const std::string& name,
                                    const std::vector<int64_t>& default_val,
                                    const std::string& description) const;
};

#define REGISTER_ENV_CONFIG_DEF() REGISTER_CONFIG_DEF(kEnvConfigDefType)
#define REGISTER_SESSION_CONFIG_DEF() REGISTER_CONFIG_DEF(kSessionConfigDefType)
#define REGISTER_FUNCTION_CONFIG_DEF() REGISTER_CONFIG_DEF(kFunctionConfigDefType)
#define REGISTER_SCOPE_CONFIG_DEF() REGISTER_CONFIG_DEF(kScopeConfigDefType)

#define REGISTER_CONFIG_DEF(config_def_type)                                                    \
  static ConfigDefBuidler<config_def_type> OF_PP_CAT(g_##config_def_type##_def_, __COUNTER__) = \
      ConfigDefBuidler<config_def_type>()

const ConfigDef& GlobalEnvConfigDef();
const ConfigDef& GlobalSessionConfigDef();
const ConfigDef& GlobalFunctionConfigDef();
const ConfigDef& GlobalScopeConfigDef();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CONFIG_DEF_H_
