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

#include <memory>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace py = pybind11;

namespace oneflow {

namespace autograd {

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  py::class_<NoGradGuard, std::shared_ptr<NoGradGuard>>(m, "no_grad")
      .def(py::init([]() { return std::make_shared<NoGradGuard>(); }))
      .def("__enter__", [](const NoGradGuard& no_grad_obj) {})
      .def("__exit__", [](const NoGradGuard& no_grad_obj, const py::object& type,
                          const py::object& value, const py::object& traceback) {});
}

}  // namespace autograd

}  // namespace oneflow
