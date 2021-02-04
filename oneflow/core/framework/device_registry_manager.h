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
#ifndef ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTRY_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTRY_MANAGER_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/device_registry.h"

namespace oneflow {

class DeviceRegistryMgr final {
 private:
  DeviceRegistryMgr() {}

 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceRegistryMgr);

 public:
  static DeviceRegistryMgr& Get();
  HashMap<DeviceType, DumpVersionInfoFn, std::hash<int>>& DumpVersionInfoFuncs();
  HashMap<DeviceType, std::string, std::hash<int>>& DeviceType2TagPair();
  HashMap<std::string, DeviceType, std::hash<std::string>>& DeviceTag2TypePair();

 private:
  HashMap<DeviceType, DumpVersionInfoFn, std::hash<int>> dump_version_info_funcs_;
  HashMap<DeviceType, std::string, std::hash<int>> device_type_to_tag_pairs_;
  HashMap<std::string, DeviceType, std::hash<std::string>> device_tag_to_type_pairs_;
};

#define REGISTER_DEVICE(device_type)                                           \
  static ::oneflow::DeviceRegistry OF_PP_CAT(g_device_registry, __COUNTER__) = \
      DeviceRegistry(device_type)

}  // namespace oneflow
#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTRY_MANAGER_H_
