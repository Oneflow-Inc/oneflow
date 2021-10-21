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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<one::Generator, std::shared_ptr<one::Generator>>(m, "Generator")
      .def(py::init([](const std::string& device_tag) {
        std::string device_name = "";
        int device_index = -1;
        ParsingDeviceTag(device_tag, &device_name, &device_index).GetOrThrow();
        return one::MakeGenerator(device_name, device_index).GetPtrOrThrow();
      }))
      .def("manual_seed", &one::Generator::set_current_seed)
      .def("initial_seed", &one::Generator::current_seed)
      .def("seed", &one::Generator::seed)
      .def_property_readonly(
          "device", [](const one::Generator& generator) { return generator.device().GetOrThrow(); })
      .def("get_state",
           [](const one::Generator& generator) { return generator.GetState().GetPtrOrThrow(); })
      .def("set_state", [](one::Generator& generator, const std::shared_ptr<one::Tensor>& state) {
        return generator.SetState(state).GetOrThrow();
      });

  m.def("manual_seed", [](uint64_t seed) { return one::ManualSeed(seed).GetOrThrow(); });
  m.def("create_generator", [](const std::string& device_tag) {
    std::string device_name = "";
    int device_index = -1;
    ParsingDeviceTag(device_tag, &device_name, &device_index).GetOrThrow();
    return one::MakeGenerator(device_name, device_index).GetPtrOrThrow();
  });
  m.def("default_generator", [](const std::string& device_tag) {
    std::string device_name = "";
    int device_index = -1;
    ParsingDeviceTag(device_tag, &device_name, &device_index).GetOrThrow();
    return one::DefaultGenerator(device_name, device_index).GetPtrOrThrow();
  });
}

}  // namespace oneflow
