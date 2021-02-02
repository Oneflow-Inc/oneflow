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
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {

DeviceRegistryMgr& DeviceRegistryMgr::Get() {
  static DeviceRegistryMgr mgr;
  return mgr;
}

HashMap<DeviceType, DumpVersionInfoFn>& DeviceRegistryMgr::DumpVersionInfoFuncs() {
  return dump_version_info_funcs_;
}

HashMap<DeviceType, GetStringFn>& DeviceRegistryMgr::StringFuncs() { return get_string_funcs_; }

std::string DeviceRegistryMgr::GetDeviceString(DeviceType dev_type) {
  if (get_string_funcs_.find(dev_type) == get_string_funcs_.end()) { UNIMPLEMENTED(); }
  return get_string_funcs_[dev_type]();
}

}  // namespace oneflow
