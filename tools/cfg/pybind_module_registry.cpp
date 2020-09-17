#include "oneflow/cfg/pybind_module_registry.h"

namespace oneflow {

namespace cfg {

std::map<std::string, std::function<void(pybind11::module&)>> Pybind11ModuleRegistry::sub_module_;

void Pybind11ModuleRegistry::Register(std::string module_path, 
                                      std::function<void(pybind11::module&)> build_sub_module) {
  CHECK(sub_module_.emplace(module_path, build_sub_module).second) << "Registered failed";
}

void Pybind11ModuleRegistry::ImportAll(pybind11::module& m) {
  for (auto& pair : sub_module_) {
    BuildSubModule(pair.first, m, pair.second);
  }
}

void Pybind11ModuleRegistry::BuildSubModule(const std::string& module_path,  pybind11::module& m,
                    const std::function<void(pybind11::module&)>& build_sub_module) {
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

} // namespace cfg

} // namespace oneflow
