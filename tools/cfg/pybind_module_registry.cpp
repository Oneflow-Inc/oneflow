#include "oneflow/cfg/pybind_module_registry.h"

namespace oneflow {

namespace cfg {

namespace {

// If different APIs are registered under the same path, the BuildModuleFuntion of which will be
// saved in the corresponding vector.
using SubModuleMap = std::map<std::string, std::vector<std::function<void(pybind11::module&)>>>;

SubModuleMap* GetSubModuleMap() {
  static SubModuleMap sub_module_map;
  return &sub_module_map;
}

}

void Pybind11ModuleRegistry::Register(std::string module_path,
                                     std::function<void(pybind11::module&)> BuildModule) {
  (*GetSubModuleMap())[module_path].emplace_back(BuildModule);
}

void Pybind11ModuleRegistry::ImportAll(pybind11::module& m) {
  for (auto& pair : (*GetSubModuleMap())) {
    for (auto& BuildModule : pair.second) { BuildSubModule(pair.first, m, BuildModule); }
  }
}

void Pybind11ModuleRegistry::BuildSubModule(
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

} // namespace cfg

} // namespace oneflow
