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
#include "oneflow/core/framework/attr.h"

namespace oneflow {

enum ConfigDefType {
  kEnvAttrDefType = 1,
  kSessionAttrDefType = 2,
  kFunctionAttrDefType = 3,
  kScopeAttrDefType = 4,
  kConstantAttrDefType = 5,
};

struct ConfigConstant final {
  static bool Bool(const std::string& name);
  static int64_t Int64(const std::string& name);
  static double Double(const std::string& name);
  static const std::string& String(const std::string& name);
};

#define REGISTER_ENV_CONFIG_DEF() REGISTER_CONFIG_DEF(kEnvAttrDefType)
#define REGISTER_SESSION_CONFIG_DEF() REGISTER_CONFIG_DEF(kSessionAttrDefType)
#define REGISTER_FUNCTION_CONFIG_DEF() REGISTER_CONFIG_DEF(kFunctionAttrDefType)
#define REGISTER_SCOPE_CONFIG_DEF() REGISTER_CONFIG_DEF(kScopeAttrDefType)
#define DEFINE_CONFIG_CONSTANT() REGISTER_CONFIG_DEF(kConstantAttrDefType)

#define REGISTER_CONFIG_DEF(config_def_type)                                      \
  static AttrDefsMutAccessor OF_PP_CAT(g_##config_def_type##_def_, __COUNTER__) = \
      AttrDefsMutAccessor(GlobalConfigDefMutAccessor<config_def_type>())

template<ConfigDefType config_def_type>
const AttrDefsAccessor& GlobalConfigDefAccessor();

template<ConfigDefType config_def_type>
const AttrDefsMutAccessor& GlobalConfigDefMutAccessor();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CONFIG_DEF_H_
