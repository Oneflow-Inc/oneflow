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
#include "oneflow/core/vm/id_generator.h"

namespace oneflow {
namespace vm {

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("vm", m) {
  py::class_<IdGenerator, std::shared_ptr<IdGenerator>>(m, "IdGenerator");
  py::class_<PhysicalIdGenerator, IdGenerator, std::shared_ptr<PhysicalIdGenerator>>(
      m, "PhysicalIdGenerator")
      .def(py::init<>())
      .def("NewSymbolId",
           [](const std::shared_ptr<PhysicalIdGenerator>& x) {
             return x->NewSymbolId().GetOrThrow();
           })
      .def("NewObjectId", [](const std::shared_ptr<PhysicalIdGenerator>& x) {
        return x->NewObjectId().GetOrThrow();
      });

  py::class_<LogicalIdGenerator, IdGenerator, std::shared_ptr<LogicalIdGenerator>>(
      m, "LogicalIdGenerator")
      .def(py::init<>())
      .def("NewSymbolId",
           [](const std::shared_ptr<LogicalIdGenerator>& x) {
             return x->NewSymbolId().GetOrThrow();
           })
      .def("NewObjectId", [](const std::shared_ptr<LogicalIdGenerator>& x) {
        return x->NewObjectId().GetOrThrow();
      });
}

}  // namespace vm
}  // namespace oneflow
