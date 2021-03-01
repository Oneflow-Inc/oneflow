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
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpBuilder, std::shared_ptr<one::OpBuilder>>(m, "OpBuilder")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Name", &one::OpBuilder::Name)
      .def("Op", &one::OpBuilder::Op)
      .def("Input",
           [](const std::shared_ptr<one::OpBuilder>& builder, const std::string& input_name,
              const int input_num) { builder->Input(input_name, input_num); })
      .def("Output",
           [](const std::shared_ptr<one::OpBuilder>& builder, const std::string& output_name,
              const int output_num) { builder->Output(output_name, output_num); })
      .def("Attr", [](const std::shared_ptr<one::OpBuilder>& builder, const std::string& attr_name,
                      const std::string& attr_value) { builder->Attr(attr_name, attr_value); })
      .def("Build", [](const std::shared_ptr<one::OpBuilder>& builder) {
        return builder->Build().GetPtrOrThrow();
      });
}

}  // namespace oneflow
