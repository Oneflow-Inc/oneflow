#ifndef CFG_PYBIND_REGISTRY_H_
#define CFG_PYBIND_REGISTRY_H_
#include <pybind11/pybind11.h>
#include <map>
#include <typeindex>
#include <vector>
#include <set>
#include <functional>

namespace oneflow {
namespace cfg {

class Pybind11Context final {
 public:
  Pybind11Context() = default;
  ~Pybind11Context() = default;

  bool IsTypeIndexRegistered(const std::type_index& type_index) {
    return GetRegisteredTypeIndices()->count(type_index) > 0;
  }

  void RegisterTypeIndex(const std::type_index& type_index) {
    GetRegisteredTypeIndices()->insert(type_index);
  }

 private:
  std::set<std::type_index>* GetRegisteredTypeIndices();
};



class Pybind11ModuleRegistry {
 public:
  Pybind11ModuleRegistry() = default;
  ~Pybind11ModuleRegistry() = default;

  void Register(std::string module_path, std::function<void(pybind11::module&)> BuildModule);
  void ImportAll(pybind11::module& m);

 private:
  void BuildSubModule(const std::string& module_path, pybind11::module& m,
                      const std::function<void(pybind11::module&)>& BuildModule);
};

} // namespace cfg

} // namespace oneflow

#define ONEFLOW_CFG_PYBIND11_MODULE(module_path, m)                  \
  static void OneflowCfgPythonModule##__LINE__(pybind11::module& m, ::oneflow::cfg::Pybind11Context* ctx);   \
  namespace {                                                        \
    void OneflowCfgPythonModule(pybind11::module& m) {               \
      ::oneflow::cfg::Pybind11Context ctx;                           \
      OneflowCfgPythonModule##__LINE__(m, &ctx);                     \
    }                                                                \
  struct CfgRegistryInit {                                           \
    CfgRegistryInit() {                                              \
        ::oneflow::cfg::Pybind11ModuleRegistry()                     \
          .Register(module_path, &OneflowCfgPythonModule);           \
    }                                                                \
  };                                                                 \
  CfgRegistryInit cfg_registry_init;                                 \
  }                                                                  \
  static void OneflowCfgPythonModule##__LINE__(pybind11::module& m, ::oneflow::cfg::Pybind11Context* ctx)

#endif // CFG_PYBIND_REGISTRY_H_
