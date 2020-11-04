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
AttrDefCollection* MutGlobalConfigDef() {
  static AttrDefCollection attr_defs;
  return &attr_defs;
}

template<ConfigDefType config_def_type>
const AttrDefCollection& GlobalConfigDef() {
  return *MutGlobalConfigDef<config_def_type>();
}

}  // namespace

bool ConfigConstant::Bool(const std::string& name) {
  return CHECK_JUST(GlobalConfigDefAccessor<kConstantAttrDefType>().Bool(name));
}

int64_t ConfigConstant::Int64(const std::string& name) {
  return CHECK_JUST(GlobalConfigDefAccessor<kConstantAttrDefType>().Int64(name));
}

double ConfigConstant::Double(const std::string& name) {
  return CHECK_JUST(GlobalConfigDefAccessor<kConstantAttrDefType>().Double(name));
}

const std::string& ConfigConstant::String(const std::string& name) {
  return CHECK_JUST(GlobalConfigDefAccessor<kConstantAttrDefType>().String(name));
}

template<ConfigDefType config_def_type>
const AttrDefsAccessor& GlobalConfigDefAccessor() {
  static const AttrDefsAccessor accessor(GlobalConfigDef<config_def_type>());
  return accessor;
}
template const AttrDefsAccessor& GlobalConfigDefAccessor<kEnvAttrDefType>();
template const AttrDefsAccessor& GlobalConfigDefAccessor<kSessionAttrDefType>();
template const AttrDefsAccessor& GlobalConfigDefAccessor<kFunctionAttrDefType>();
template const AttrDefsAccessor& GlobalConfigDefAccessor<kScopeAttrDefType>();
template const AttrDefsAccessor& GlobalConfigDefAccessor<kConstantAttrDefType>();

template<ConfigDefType config_def_type>
const AttrDefsMutAccessor& GlobalConfigDefMutAccessor() {
  static const AttrDefsMutAccessor accessor(MutGlobalConfigDef<config_def_type>());
  return accessor;
}
template const AttrDefsMutAccessor& GlobalConfigDefMutAccessor<kEnvAttrDefType>();
template const AttrDefsMutAccessor& GlobalConfigDefMutAccessor<kSessionAttrDefType>();
template const AttrDefsMutAccessor& GlobalConfigDefMutAccessor<kFunctionAttrDefType>();
template const AttrDefsMutAccessor& GlobalConfigDefMutAccessor<kScopeAttrDefType>();
template const AttrDefsMutAccessor& GlobalConfigDefMutAccessor<kConstantAttrDefType>();

}  // namespace oneflow
