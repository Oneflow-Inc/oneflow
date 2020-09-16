#ifndef CFG_PYBIND_REGISTRY_H_
#define CFG_PYBIND_REGISTRY_H_
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <map>

namespace oneflow {
namespace cfg {

class Pybind11ModuleRegistry {
 public:
  Pybind11ModuleRegistry() = default;
  ~Pybind11ModuleRegistry() = default;
  void Register(std::string module_path, std::function<void(pybind11::module&)> build_sub_module);
  void ImportAll(pybind11::module& m);
 private:
  void BuildSubModule(const std::string& module_path,  pybind11::module& m,
                    const std::function<void(pybind11::module&)>& build_sub_module);
  static std::map<std::string, std::function<void(pybind11::module&)>> sub_module_;
};

} // namespace cfg

} // namespace oneflow


#define ONEFLOW_PYBIND11_MODULE(module_path, m)                   \
  static void OneflowPythonModule##__LINE__(pybind11::module&);    \
  namespace {                                                     \
  struct RegistryInit {                                           \
    RegistryInit() {                                              \
        ::oneflow::cfg::Pybind11ModuleRegistry()                  \
          .Register(module_path, &OneflowPythonModule##__LINE__);  \
    }                                                             \
  };                                                              \
  RegistryInit registry_init;                                     \
  }                                                               \
  static void OneflowPythonModule##__LINE__(pybind11::module& m)

#endif // CFG_PYBIND_REGISTRY_H_
