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
#include "oneflow/api/python/util/of_api_registry.h"

namespace oneflow {

std::map<std::string, std::vector<std::function<void(pybind11::module&)>>>
    OneflowModuleRegistry::sub_module_;

void OneflowModuleRegistry::Register(std::string module_path,
                                     std::function<void(pybind11::module&)> build_sub_module) {
  sub_module_[module_path].emplace_back(build_sub_module);
}

void OneflowModuleRegistry::ImportAll(pybind11::module& m) {
  for (auto& pair : sub_module_) {
    for (auto& build_sub_module : pair.second) { BuildSubModule(pair.first, m, build_sub_module); }
  }
}

void OneflowModuleRegistry::BuildSubModule(
    const std::string& module_path, pybind11::module& m,
    const std::function<void(pybind11::module&)>& build_sub_module) {
  if (module_path.empty()) {
    build_sub_module(m);
    return;
  }
  size_t dot_pos = module_path.find(".");
  if (dot_pos == std::string::npos) {
    pybind11::module sub_module = m.def_submodule((char*)module_path.data());
    build_sub_module(sub_module);
  } else {
    const std::string& sub_module_name = module_path.substr(0, dot_pos);
    pybind11::module sub_module = m.def_submodule((char*)sub_module_name.data());
    BuildSubModule(module_path.substr(dot_pos + 1), sub_module, build_sub_module);
  }
}

}  // namespace oneflow
