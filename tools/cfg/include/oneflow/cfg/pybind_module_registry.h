#ifndef CFG_PYBIND_REGISTRY_H_
#define CFG_PYBIND_REGISTRY_H_
#include <pybind11/pybind11.h>
#include <map>
#include <vector>
#include <functional>

namespace oneflow {
namespace cfg {

class Pybind11ModuleRegistry {
 public:
  Pybind11ModuleRegistry() = default;
  ~Pybind11ModuleRegistry() = default;
  
  void Register(std::string module_path, std::function<void(pybind11::module&)> BuildModule);
  void ImportAll(pybind11::module& m);

 private:
  void BuildSubModule(const std::string& module_path, pybind11::module& m,
                      const std::function<void(pybind11::module&)>& BuildModule);
  // If different APIs are registered under the same path, the BuildModuleFuntion of which will be
  // saved in the corresponding vector.
  using SubModuleMap = std::map<std::string, std::vector<std::function<void(pybind11::module&)>>>;
  static SubModuleMap sub_module_map_;
};

} // namespace cfg

} // namespace oneflow


#define ONEFLOW_CFG_PYBIND11_MODULE(module_path, m)                  \
  static void OneflowCfgPythonModule##__LINE__(pybind11::module&);   \
  namespace {                                                        \
  struct CfgRegistryInit {                                           \
    CfgRegistryInit() {                                              \
        ::oneflow::cfg::Pybind11ModuleRegistry()                     \
          .Register(module_path, &OneflowCfgPythonModule##__LINE__); \
    }                                                                \
  };                                                                 \
  CfgRegistryInit cfg_registry_init;                                 \
  }                                                                  \
  static void OneflowCfgPythonModule##__LINE__(pybind11::module& m)

#endif // CFG_PYBIND_REGISTRY_H_
