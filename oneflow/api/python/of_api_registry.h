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
#ifndef ONEFLOW_API_PYTHON_UTIL_OF_API_REGISTRY_H_
#define ONEFLOW_API_PYTHON_UTIL_OF_API_REGISTRY_H_
#include <pybind11/pybind11.h>
#include <map>
#include <vector>
#include <functional>

namespace oneflow {

class OneflowModuleRegistry {
 public:
  OneflowModuleRegistry() = default;
  ~OneflowModuleRegistry() = default;

  void Register(std::string module_path, std::function<void(pybind11::module&)> BuildModule);
  void ImportAll(pybind11::module& m);

 private:
  void BuildSubModule(const std::string& module_path, pybind11::module& m,
                      const std::function<void(pybind11::module&)>& BuildModule);
};

}  // namespace oneflow

#define ONEFLOW_API_PYBIND11_MODULE(module_path, m)                                                \
  static void OneflowApiPythonModule##__LINE__(pybind11::module&);                                 \
  namespace {                                                                                      \
  struct OfApiRegistryInit {                                                                       \
    OfApiRegistryInit() {                                                                          \
      ::oneflow::OneflowModuleRegistry().Register(module_path, &OneflowApiPythonModule##__LINE__); \
    }                                                                                              \
  };                                                                                               \
  OfApiRegistryInit of_api_registry_init;                                                          \
  }                                                                                                \
  static void OneflowApiPythonModule##__LINE__(pybind11::module& m)

#endif  // ONEFLOW_API_PYTHON_UTIL_OF_API_REGISTRY_H_
