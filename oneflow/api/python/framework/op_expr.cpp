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

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr");

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name);

  py::class_<one::UserOpExpr, one::BuiltinOpExpr, std::shared_ptr<one::UserOpExpr>>(m, "UserOpExpr")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def_property_readonly("type", &one::UserOpExpr::type)
      .def_property_readonly(
          "proto", [](const one::UserOpExpr& op) { return PbMessage2TxtString(op.proto()); })
      .def_property_readonly("indexed_input_names", &one::UserOpExpr::indexed_input_names);
}

}  // namespace oneflow
