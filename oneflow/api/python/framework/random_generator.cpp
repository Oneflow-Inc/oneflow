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
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {

Maybe<one::Generator> CreateGenerator(const std::string& device_str) {
  auto [device_name, device_index] = *JUST(ParseDeviceString(device_str));
  return one::MakeGenerator(device_name, device_index);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<one::Generator, std::shared_ptr<one::Generator>>(m, "Generator")
      .def(py::init([](const std::string& device_tag) {
        return CreateGenerator(device_tag).GetPtrOrThrow();
      }))
      .def("manual_seed",
           [](const std::shared_ptr<one::Generator>& generator,
              const py::object& seed) -> std::shared_ptr<one::Generator> {
             int64_t seed_val = (one::functional::PyUnpackLong(seed.ptr())).GetOrThrow();
             generator->set_current_seed(seed_val);
             return generator;
           })
      .def("initial_seed", &one::Generator::current_seed)
      .def("seed", &one::Generator::seed)
      .def_property_readonly("device", &one::Generator::device)
      .def("get_state", &one::Generator::GetState)
      .def("set_state", &one::Generator::SetState);

  m.def("manual_seed", [](const py::object& seed) -> Maybe<one::Generator> {
    int64_t seed_val = JUST(one::functional::PyUnpackLong(seed.ptr()));
    return one::ManualSeed(seed_val);
  });
  m.def("manual_seed",
        [](const py::object& seed, const std::string& device, int device_index) -> Maybe<void> {
          int64_t seed_val = JUST(one::functional::PyUnpackLong(seed.ptr()));
          return one::ManualSeed(seed_val, device, device_index);
        });
  m.def("create_generator", &CreateGenerator);
  m.def("default_generator", [](const std::string& device_str) -> Maybe<one::Generator> {
    auto [device_name, device_index] = *JUST(ParseDeviceString(device_str));
    return one::DefaultGenerator(device_name, device_index);
  });
  m.def("ManualSeedAllCudaGenerator", [](const py::object& seed) -> Maybe<void> {
    int64_t seed_val = JUST(one::functional::PyUnpackLong(seed.ptr()));
    return one::ManualSeedAllCudaGenerator(seed_val);
  });
}

}  // namespace oneflow
