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
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/op_builder.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::Operation, std::shared_ptr<one::Operation>>(m, "Operation")
      .def(py::init<>())
      .def_readonly("name", &one::Operation::op_name)
      .def_readonly("proto", &one::Operation::proto)
      .def_readonly("indexed_input_names", &one::Operation::indexed_input_names);

  py::class_<one::OpBuilder, std::shared_ptr<one::OpBuilder>>(m, "OpBuilder")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Name", &one::OpBuilder::Name)
      .def("Op", &one::OpBuilder::Op)
      .def("Input", &one::OpBuilder::Input)
      .def("Output", py::overload_cast<const std::string&>(&one::OpBuilder::Output))
      .def("Output", py::overload_cast<const std::string&, const int>(&one::OpBuilder::Output))
      .def("Attr", &one::OpBuilder::Attr)
      .def("Build", &one::OpBuilder::Build);
}

}  // namespace oneflow
