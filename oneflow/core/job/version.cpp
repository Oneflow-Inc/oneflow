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
#include "oneflow/core/job/version.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {

void DumpVersionInfo() {
#ifdef WITH_GIT_VERSION
  LOG(INFO) << "OneFlow git version: " << GetOneFlowGitVersion();
#endif  // WITH_GIT_VERSION
  auto dump_info_funcs = DeviceRegistryMgr::Get().DumpVersionInfoFuncs();
  for (auto dev_func_pair : dump_info_funcs) { dev_func_pair.second(); }
}

}  // namespace oneflow
