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
#include "oneflow/api/python/of_api_registry.h"

namespace oneflow {

namespace {

// If different APIs are registered under the same path, the BuildModuleFuntion of which will be
// saved in the corresponding vector.
using SubModuleMap = std::map<std::string, std::vector<std::function<void(pybind11::module&)>>>;

SubModuleMap* GetSubModuleMap() {
  static SubModuleMap sub_module_map;
  return &sub_module_map;
}

}  // namespace

void OneflowModuleRegistry::Register(std::string module_path,
                                     std::function<void(pybind11::module&)> BuildModule) {
  (*GetSubModuleMap())[module_path].emplace_back(BuildModule);
}

void OneflowModuleRegistry::ImportAll(pybind11::module& m) {
  for (const auto& pair : (*GetSubModuleMap())) {
    for (const auto& BuildModule : pair.second) { BuildSubModule(pair.first, m, BuildModule); }
  }
}

void OneflowModuleRegistry::BuildSubModule(
    const std::string& module_path, pybind11::module& m,
    const std::function<void(pybind11::module&)>& BuildModule) {
  if (module_path.empty()) {
    BuildModule(m);
    return;
  }
  size_t dot_pos = module_path.find(".");
  if (dot_pos == std::string::npos) {
    pybind11::module sub_module = m.def_submodule(module_path.data());
    BuildModule(sub_module);
  } else {
    const std::string& sub_module_name = module_path.substr(0, dot_pos);
    pybind11::module sub_module = m.def_submodule(sub_module_name.data());
    BuildSubModule(module_path.substr(dot_pos + 1), sub_module, BuildModule);
  }
}

}  // namespace oneflow
